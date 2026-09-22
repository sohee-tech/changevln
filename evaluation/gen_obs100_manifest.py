#!/usr/bin/env python3
"""
100-episode obstacle manifest generator.

Uses the same 100 scene-grouped val_unseen episodes as the LSE experiment.
For each episode, auto-places a 0.4×1.2×0.4m box obstacle on the reference
path so that detour_ratio is in [1.05, 3.0].

Output: eval_out/obs100_manifest.json
  [{"episode_id": int, "x": float, "z": float, "y": float,
    "yaw_deg": 0.0, "detour_ratio": float, "geo_base": float, "geo_obs": float}, ...]

Episodes that cannot be placed are recorded with "placed": false.
"""

import gzip, json, logging, math, os, sys, csv
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
os.chdir(Path(__file__).parent.parent.resolve())

import habitat_sim
import magnum as mn
from habitat_sim.nav import ShortestPath

from habitat_extensions.obstacle_injector import ObstacleInjector, ObstacleSpec

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("gen_obs100")

DATASET_GZ = "data/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz"
GT_GZ      = "data/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen_gt.json.gz"
SCENE_ROOT = "data/scene_datasets/"
OUT_PATH   = "eval_out/obs100_manifest.json"

BOX_SIZE   = (0.4, 1.2, 0.4)

# The 100 scene-grouped val_unseen episode IDs from the LSE experiment.
LSE_100_IDS = [
    1,2,3,13,14,15,25,26,27,37,38,39,40,41,42,52,53,54,55,56,57,
    73,74,75,85,86,87,124,125,126,136,137,138,145,146,147,154,155,156,
    163,164,165,235,236,237,247,248,249,250,251,252,253,254,255,
    292,293,294,301,302,303,352,353,354,361,362,363,367,368,369,
    385,386,387,415,416,417,418,419,420,439,440,441,445,446,447,
    475,476,477,505,506,507,514,515,516,544,545,546,571,572,573,628
]
assert len(LSE_100_IDS) == 100, f"Expected 100 IDs, got {len(LSE_100_IDS)}"


def geodesic(pf, start, goal):
    sp = ShortestPath()
    sp.requested_start = mn.Vector3(*[float(v) for v in start])
    sp.requested_end   = mn.Vector3(*[float(v) for v in goal])
    return float(sp.geodesic_distance) if pf.find_path(sp) else float("inf")


def build_sim(scene_glb: str):
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_id = scene_glb
    cfg.enable_physics = True
    cfg.gpu_device_id = 0
    rgb = habitat_sim.SensorSpec()
    rgb.uuid = "rgb"; rgb.sensor_type = habitat_sim.SensorType.COLOR
    rgb.resolution = [32, 32]; rgb.position = [0, 1.25, 0]
    rgb.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    agent = habitat_sim.agent.AgentConfiguration()
    agent.sensor_specifications = [rgb]
    return habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent]))


def try_place(injector, waypoints, ratio, spread, sign, start, goal, floor_y, pf):
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
        geo_obs = geodesic(pf, start, goal)
        return obj_id, pos.tolist(), geo_obs
    return -1, None, None


def place_for_episode(ep_data, gt_data, sim, injector):
    ep_id = ep_data["episode_id"]
    start = np.array(ep_data["start_position"], dtype=float)
    goal  = np.array(ep_data["goals"][0]["position"], dtype=float)
    floor_y = float(start[1])
    pf = sim.pathfinder

    waypoints = gt_data.get(str(ep_id), gt_data.get(ep_id, {})).get("locations", [])
    if not waypoints:
        waypoints = ep_data.get("reference_path", [ep_data["start_position"], ep_data["goals"][0]["position"]])

    geo_base = geodesic(pf, start, goal)
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
                    injector, waypoints, ratio, spread, sign, start, goal, floor_y, pf)
                if obj_id >= 0:
                    det = geo_obs / geo_base
                    if 1.05 <= det <= 3.0:
                        candidate = {
                            "episode_id": ep_id,
                            "placed": True,
                            "x": round(pos[0], 4),
                            "z": round(pos[2], 4),
                            "y": round(pos[1], 4),
                            "yaw_deg": 0.0,
                            "geo_base": round(geo_base, 3),
                            "geo_obs": round(geo_obs, 3),
                            "detour_ratio": round(det, 3),
                        }
                        if best is None or abs(det - 1.3) < abs(best["detour_ratio"] - 1.3):
                            best = candidate
                    injector._sim.remove_object(obj_id)
                    if obj_id in injector._injected:
                        injector._injected.remove(obj_id)
                    injector._recompute_navmesh()

    return best


def main():
    print("Loading dataset...")
    with gzip.open(DATASET_GZ) as f:
        ds = json.load(f)
    with gzip.open(GT_GZ) as f:
        gt = json.load(f)

    ep_map = {ep["episode_id"]: ep for ep in ds["episodes"]}
    target_set = set(LSE_100_IDS)
    target_eps = [ep_map[eid] for eid in LSE_100_IDS if eid in ep_map]
    missing = [eid for eid in LSE_100_IDS if eid not in ep_map]
    if missing:
        print(f"WARNING: {len(missing)} episode IDs not found in val_unseen: {missing}")

    # Group by scene
    scene_eps = defaultdict(list)
    for ep in target_eps:
        scene_eps[ep["scene_id"]].append(ep)
    print(f"Target: {len(target_eps)} episodes across {len(scene_eps)} scenes")

    results = {}
    total_placed = 0

    for scene_id, eps in sorted(scene_eps.items()):
        scene_glb = os.path.join(SCENE_ROOT, scene_id)
        print(f"\nScene: {scene_id} ({len(eps)} eps)")
        sim = build_sim(scene_glb)
        injector = ObstacleInjector(sim, recompute_navmesh=True)

        for ep in eps:
            ep_id = ep["episode_id"]
            result = place_for_episode(ep, gt, sim, injector)
            if result is not None:
                results[ep_id] = result
                total_placed += 1
                print(f"  ep={ep_id}: placed at ({result['x']:.3f}, {result['z']:.3f})"
                      f"  detour={result['detour_ratio']:.3f}")
            else:
                results[ep_id] = {
                    "episode_id": ep_id,
                    "placed": False,
                    "x": None, "z": None, "y": None, "yaw_deg": 0.0,
                    "geo_base": None, "geo_obs": None, "detour_ratio": None,
                }
                print(f"  ep={ep_id}: FAILED to place obstacle")

        sim.close()

    # Write manifest (in LSE episode order)
    manifest = [results[eid] for eid in LSE_100_IDS if eid in results]
    os.makedirs("eval_out", exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    placed = [r for r in manifest if r["placed"]]
    failed = [r for r in manifest if not r["placed"]]
    print(f"\n=== Manifest saved: {OUT_PATH}")
    print(f"  Placed: {len(placed)}/100")
    print(f"  Failed: {len(failed)}/100")
    if failed:
        print(f"  Failed IDs: {[r['episode_id'] for r in failed]}")
    if placed:
        ratios = [r["detour_ratio"] for r in placed]
        print(f"  detour_ratio: min={min(ratios):.3f}  max={max(ratios):.3f}  "
              f"mean={sum(ratios)/len(ratios):.3f}")


if __name__ == "__main__":
    main()
