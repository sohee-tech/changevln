"""
gen_benchmark_manifest.py
==========================
Auto-generate obstacle benchmark manifest for all val_unseen episodes.

Reuses gen_obs100_manifest.py obstacle-placement logic unchanged.
Adds region metadata (instruction, ordered region sequence, AABB per category)
as auxiliary info for future instruction-aware evaluation.

Region constraint does NOT gate episode inclusion:
  geometry-only benchmark is primary; region info is optional metadata.

Output fields per episode:
  placed, x/z/y, yaw_deg, geo_base, geo_obs, detour_ratio, scene_id,
  instruction, region_sequence, region_aabbs, has_region_constraint

Crash-safe: progress written to PROGRESS_LOG (jsonl) after each episode.
Resume: already-done episodes loaded from PROGRESS_LOG on restart.

Usage:
  python gen_benchmark_manifest.py            # full 1839 episodes
  python gen_benchmark_manifest.py --smoke    # first 20 episodes only
  python gen_benchmark_manifest.py --n 100   # first N episodes
"""

import argparse
import gzip
import json
import logging
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.resolve()))
os.chdir(Path(__file__).parent.resolve())

import habitat_sim
import magnum as mn
from habitat_sim.nav import ShortestPath

from habitat_extensions.obstacle_injector import ObstacleInjector, ObstacleSpec

logging.basicConfig(level=logging.WARNING)

# ═══════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════
DATASET_GZ   = "data/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz"
GT_GZ        = "data/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen_gt.json.gz"
SCENE_ROOT   = "data/scene_datasets"
OUT_JSON     = "eval_out/benchmark_manifest.json"
PROGRESS_LOG = "eval_out/benchmark_manifest_progress.jsonl"

BOX_SIZE     = (0.4, 1.2, 0.4)

# ═══════════════════════════════════════════════════════════════════════
# REGION DEFINITIONS  (identical to region pilot)
# ═══════════════════════════════════════════════════════════════════════

HOUSE_CAT = {
    'k': 'kitchen',    'h': 'hallway',    'b': 'bedroom',   't': 'toilet',
    'c': 'closet',     'a': 'living room','d': 'dining room','l': 'laundry',
    'j': 'staircase',  'e': 'entrance',   'f': 'family room',
    # 'p'=porch excluded (pantry→porch mapping unreliable)
}

INSTR_TO_REGION = [
    ('galley kitchen',  'kitchen'),
    ('dining room',     'dining room'),
    ('living room',     'living room'),
    ('family room',     'family room'),
    ('lounge chair',    None),          # compound — exclude
    ('lounge room',     'living room'),
    ('bathroom',        'toilet'),
    ('restroom',        'toilet'),
    ('hallway',         'hallway'),
    ('corridor',        'hallway'),
    ('bedroom',         'bedroom'),
    ('kitchen',         'kitchen'),
    ('closet',          'closet'),
    ('laundry',         'laundry'),
    ('staircase',       'staircase'),
    ('stairs',          'staircase'),
    ('entrance',        'entrance'),
    ('entryway',        'entrance'),
    ('hall',            'hallway'),
]


def parse_house_regions(house_file: str) -> Dict[str, List[dict]]:
    """Parse .house → {cat_name: [{min_x, max_x, min_z, max_z, region_id}, ...]}
    Coordinate mapping: .house(x, y_floor) → Habitat(x, z=-y_floor)
    """
    result = defaultdict(list)
    with open(house_file) as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0] != 'R':
                continue
            cat_code = parts[5]
            if cat_code not in HOUSE_CAT:
                continue
            cat_name = HOUSE_CAT[cat_code]
            region_id = int(parts[1])
            min_hx, min_hy = float(parts[9]),  float(parts[10])
            max_hx, max_hy = float(parts[12]), float(parts[13])
            result[cat_name].append({
                'region_id': region_id,
                'min_x': min_hx, 'max_x': max_hx,
                'min_z': -max_hy, 'max_z': -min_hy,   # .house y → Habitat z=-y
            })
    return dict(result)


def extract_ordered_regions(text: str,
                            region_db: Dict) -> Tuple[List[str], Dict[str, List[dict]]]:
    """
    Returns:
      seq   : ordered list of region names (consecutive deduped)
      aabbs : {region_name: [aabb_list]} for matched regions only
    """
    text_lo = text.lower()
    hits = []
    consumed = set()

    for kw, region_name in INSTR_TO_REGION:
        idx = 0
        while True:
            pos = text_lo.find(kw, idx)
            if pos < 0:
                break
            span = range(pos, pos + len(kw))
            if any(p in consumed for p in span):
                idx = pos + 1
                continue
            if region_name is not None and region_name in region_db:
                hits.append((pos, region_name))
                for p in span:
                    consumed.add(p)
            elif region_name is None:
                for p in span:
                    consumed.add(p)
            idx = pos + 1

    if not hits:
        return [], {}

    hits.sort(key=lambda h: h[0])

    seq = []
    prev = None
    for _, cat in hits:
        if cat != prev:
            seq.append(cat)
            prev = cat

    aabbs = {cat: region_db[cat] for cat in seq if cat in region_db}
    return seq, aabbs


# ═══════════════════════════════════════════════════════════════════════
# OBSTACLE PLACEMENT  (from gen_obs100_manifest.py, unchanged logic)
# ═══════════════════════════════════════════════════════════════════════

def geodesic_dist(pf, start, goal) -> float:
    sp = ShortestPath()
    sp.requested_start = mn.Vector3(*[float(v) for v in start])
    sp.requested_end   = mn.Vector3(*[float(v) for v in goal])
    return float(sp.geodesic_distance) if pf.find_path(sp) else float("inf")


def make_sim(scene_glb: str) -> habitat_sim.Simulator:
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_id       = scene_glb
    cfg.enable_physics = True
    cfg.gpu_device_id  = 0
    cfg.create_renderer = False     # no renderer needed for placement
    agent = habitat_sim.agent.AgentConfiguration()
    return habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent]))


def try_place(injector, waypoints, ratio, spread, sign,
              start, goal, floor_y, pf):
    n = len(waypoints)
    idx = max(1, min(int(ratio * (n - 1)), n - 2))
    p0 = np.array(waypoints[idx], dtype=float)
    p1 = np.array(waypoints[min(idx + 1, n - 1)], dtype=float)
    fwd = p1 - p0; fwd[1] = 0
    norm_f = np.linalg.norm(fwd)
    if norm_f < 1e-6:
        return -1, None, None
    fwd /= norm_f
    right = np.array([-fwd[2], 0.0, fwd[0]])
    pos = p0 + sign * spread * right if spread > 0 else p0.copy()
    pos[1] = floor_y + BOX_SIZE[1] / 2.0
    wall_d = pf.distance_to_closest_obstacle(
        mn.Vector3(float(pos[0]), floor_y, float(pos[2])), max_search_radius=3.0)
    if wall_d < max(BOX_SIZE[0], BOX_SIZE[2]):
        return -1, None, None
    spec = ObstacleSpec(position=tuple(pos.tolist()), size=BOX_SIZE, yaw_deg=0.0)
    obj_id = injector._place_if_detour_exists(spec, start, goal)
    if obj_id >= 0:
        geo_obs = geodesic_dist(pf, start, goal)
        return obj_id, pos.tolist(), geo_obs
    return -1, None, None


def place_for_episode(ep_data, gt_data, sim, injector) -> Optional[dict]:
    ep_id = ep_data["episode_id"]
    start = np.array(ep_data["start_position"], dtype=float)
    goal  = np.array(ep_data["goals"][0]["position"], dtype=float)
    floor_y = float(start[1])
    pf = sim.pathfinder

    gt_rec = gt_data.get(str(ep_id), gt_data.get(ep_id, {}))
    waypoints = gt_rec.get("locations", [])
    if not waypoints:
        waypoints = ep_data.get("reference_path",
                                [ep_data["start_position"],
                                 ep_data["goals"][0]["position"]])

    geo_base = geodesic_dist(pf, start, goal)
    if geo_base == float("inf"):
        return None

    spreads = [0.0, 0.5, 1.0, 1.2, 1.6, 2.0, 2.5]
    ratios  = list(np.arange(0.15, 0.90, 0.05))

    best = None
    for ratio in ratios:
        for spread in spreads:
            signs = [0] if spread == 0 else [1, -1]
            for sign in signs:
                obj_id, pos, geo_obs = try_place(
                    injector, waypoints, ratio, spread, sign,
                    start, goal, floor_y, pf)
                if obj_id >= 0:
                    det = geo_obs / geo_base
                    if 1.05 <= det <= 3.0:
                        candidate = {
                            "x":            round(float(pos[0]), 4),
                            "z":            round(float(pos[2]), 4),
                            "y":            round(float(pos[1]), 4),
                            "yaw_deg":      0.0,
                            "geo_base":     round(geo_base, 3),
                            "geo_obs":      round(geo_obs, 3),
                            "detour_ratio": round(det, 3),
                        }
                        if best is None or abs(det - 1.3) < abs(best["detour_ratio"] - 1.3):
                            best = candidate
                    injector._sim.remove_object(obj_id)
                    if obj_id in injector._injected:
                        injector._injected.remove(obj_id)
                    injector._recompute_navmesh()
    return best


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Process first 20 episodes only (smoke test)")
    parser.add_argument("--n", type=int, default=None,
                        help="Process first N episodes")
    parser.add_argument("--out", default=OUT_JSON)
    args = parser.parse_args()

    # Progress log path derived from --out (avoids overwriting smoke test log)
    progress_log = args.out.replace(".json", "_progress.jsonl")

    t_start = time.time()
    is_smoke = args.smoke
    limit = 20 if is_smoke else args.n

    print(f"=== Benchmark Manifest Generator {'(smoke test: 20 eps)' if is_smoke else ''} ===\n")

    # ── Load dataset ─────────────────────────────────────────────────
    print("Loading dataset and GT...")
    with gzip.open(DATASET_GZ) as f:
        val = json.load(f)
    with gzip.open(GT_GZ) as f:
        gt = json.load(f)

    all_eps = val["episodes"]
    if limit:
        all_eps = all_eps[:limit]
    print(f"Target episodes: {len(all_eps)}")

    # ── Group by scene ────────────────────────────────────────────────
    scene_eps: Dict[str, List] = defaultdict(list)
    for ep in all_eps:
        scene_eps[ep["scene_id"]].append(ep)
    print(f"Scenes: {len(scene_eps)}\n")

    # ── Load progress (crash-safe resume) ────────────────────────────
    completed: Dict[str, dict] = {}
    if os.path.exists(progress_log):
        with open(progress_log) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    completed[str(rec["episode_id"])] = rec
        print(f"Resuming: {len(completed)} episodes already done\n")

    progress_f = open(progress_log, "a")
    results: Dict[str, dict] = dict(completed)

    # ── Process scene by scene ────────────────────────────────────────
    for scene_id, eps in sorted(scene_eps.items()):
        scene_name = scene_id.split("/")[1]
        house_file = os.path.join(SCENE_ROOT, scene_id.replace(".glb", ".house")
                                               .replace(scene_name+"/"+scene_name,
                                                        scene_name+"/"+scene_name))
        house_file = os.path.join(SCENE_ROOT, f"mp3d/{scene_name}/{scene_name}.house")
        scene_glb  = os.path.join(SCENE_ROOT, scene_id)

        # Parse .house regions for this scene
        region_db = parse_house_regions(house_file) if os.path.exists(house_file) else {}
        print(f"Scene: {scene_name}  eps={len(eps)}  "
              f"regions={list(region_db.keys()) if region_db else 'NO .HOUSE'}")

        # Skip if all eps in this scene already done
        pending = [ep for ep in eps if str(ep["episode_id"]) not in completed]
        if not pending:
            print(f"  All {len(eps)} eps already done, skipping sim init\n")
            continue

        sim = make_sim(scene_glb)
        injector = ObstacleInjector(sim, recompute_navmesh=True)
        n_placed = 0; n_failed = 0

        for ep in eps:
            ep_id = str(ep["episode_id"])
            if ep_id in completed:
                print(f"  ep {ep_id:>5}: SKIP (cached)")
                continue

            t_ep = time.time()
            instr_text = ep["instruction"]["instruction_text"]

            # ── Obstacle placement ────────────────────────────────
            place_result = place_for_episode(ep, gt, sim, injector)

            if place_result is not None:
                placed = True
                n_placed += 1
            else:
                placed = False
                n_failed += 1
                place_result = {"x": None, "z": None, "y": None, "yaw_deg": 0.0,
                                "geo_base": None, "geo_obs": None, "detour_ratio": None}

            # ── Region metadata ───────────────────────────────────
            if region_db:
                seq, aabbs = extract_ordered_regions(instr_text, region_db)
            else:
                seq, aabbs = [], {}

            rec = {
                "episode_id":           int(ep_id),
                "scene_id":             scene_id,
                "placed":               placed,
                **place_result,
                # Region metadata (auxiliary — does not gate inclusion)
                "instruction":          instr_text,
                "region_sequence":      seq,
                "region_aabbs":         aabbs,
                "has_region_constraint": len(seq) > 0,
            }
            results[ep_id] = rec
            progress_f.write(json.dumps(rec) + "\n")
            progress_f.flush()

            dt = time.time() - t_ep
            status = f"placed  det={place_result['detour_ratio']:.3f}" if placed else "FAILED"
            rc_str = "→".join(seq) if seq else "(none)"
            print(f"  ep {ep_id:>5}: {status:20}  regions=[{rc_str}]  [{dt:.1f}s]")

        sim.close()
        print(f"  Scene done: placed={n_placed}  failed={n_failed}\n")

    progress_f.close()

    # ── Save final JSON ───────────────────────────────────────────────
    ordered_results = [results[str(ep["episode_id"])]
                       for ep in all_eps if str(ep["episode_id"]) in results]

    os.makedirs("eval_out", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(ordered_results, f, indent=2)
    print(f"Saved: {args.out}")

    # ── Summary ────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    total      = len(ordered_results)
    n_placed   = sum(1 for r in ordered_results if r["placed"])
    n_failed   = total - n_placed
    n_with_rc  = sum(1 for r in ordered_results if r.get("has_region_constraint"))
    n_placed_rc = sum(1 for r in ordered_results if r["placed"] and r.get("has_region_constraint"))

    print(f"\n{'='*60}")
    print(f"SUMMARY  ({elapsed:.1f}s = {elapsed/60:.1f} min)")
    print(f"{'='*60}")
    print(f"  Total processed   : {total}")
    print(f"  Placed (valid)    : {n_placed} ({100*n_placed/total:.1f}%)")
    print(f"  Failed (excluded) : {n_failed}")
    if n_placed:
        ratios = [r["detour_ratio"] for r in ordered_results if r["placed"]]
        print(f"  Detour ratio      : min={min(ratios):.3f} max={max(ratios):.3f} "
              f"mean={sum(ratios)/len(ratios):.3f}")
    print(f"  Has region constraint (all)    : {n_with_rc}/{total}")
    print(f"  Has region constraint (placed) : {n_placed_rc}/{n_placed if n_placed else 1}")
    if total > 0:
        print(f"  Avg time / episode: {elapsed/total:.1f}s")
        print(f"  Est. full (1839)  : {elapsed/total*1839/60:.0f} min")


if __name__ == "__main__":
    main()
