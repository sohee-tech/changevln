"""
compute_metrics_sub500.py
=========================
500-ep benchmark metrics:  91 existing (reused) + 409 new (computed here).

Metrics per episode (Base and OAP):
  SR, PL, dSPL, nDTW, mp_ndtw_geo, GP, GP_clean, PD

Data sources:
  - 91 existing : eval_out/metrics_91ep.json  (all metrics already present)
  - 409 new obs : eval_out/sub500_base_traj_v2 / sub500_oap_traj_v2
  - 409 clean   : eval_out/sub500_base_clean_traj_v2 / sub500_oap_clean_traj_v2
  - manifest    : eval_out/sub500_manifest.json  (geo_base, geo_obs, obstacle pos)

mp_ndtw_geo: K_total=7 (1 direct + 6 via-point), seed=42, resample=0.25m.
References generated WITH obstacle placed (same as compute_metrics_91ep.py).
Grouped by scene to minimise sim-reload overhead.

Progress saved per episode → crash-safe resume.
"""

import json, gzip, csv, sys, os, time
import numpy as np
from collections import defaultdict
from typing import List, Optional, Tuple

import habitat_sim
from habitat_sim.nav import ShortestPath
import magnum as mn

sys.path.insert(0, os.path.dirname(__file__))
from habitat_extensions.obstacle_injector import ObstacleInjector, ObstacleSpec

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# ── paths ─────────────────────────────────────────────────────────────────────
SCENE_DIR    = "data/scene_datasets"
DATASET_GZ   = "data/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz"
MANIFEST_500 = "eval_out/sub500_manifest.json"
MANIFEST_91  = "eval_out/obs100_manifest_placed.json"
METRICS_91   = "eval_out/metrics_91ep.json"

RPATH = "navila-llama3-8b-8f/VLN-CE-v1/val_unseen/val_unseen_1-0.json"
BASE_OBS    = f"eval_out/sub500_base_traj_v2/{RPATH}"
BASE_CLEAN  = f"eval_out/sub500_base_clean_traj_v2/{RPATH}"
OAP_OBS     = f"eval_out/sub500_oap_traj_v2/{RPATH}"
OAP_CLEAN   = f"eval_out/sub500_oap_clean_traj_v2/{RPATH}"

OUT_JSON     = "eval_out/metrics_sub500.json"
OUT_CSV      = "eval_out/metrics_sub500.csv"
PROGRESS_LOG = "eval_out/metrics_sub500_progress.jsonl"

# ── mp_ndtw_geo config (identical to 91-ep script) ────────────────────────────
K_VIA          = 6
VIA_OVERSAMPLE = 30
VIA_SEED       = 42
RESAMPLE_STEP  = 0.25
SUCCESS_DIST   = 3.0
BOX_SIZE       = (0.4, 1.2, 0.4)
DEDUP_RATIO    = 0.10


# ── helpers (identical to compute_metrics_91ep.py) ────────────────────────────

def resample_path(points: List, step: float = RESAMPLE_STEP) -> List:
    if len(points) < 2:
        return list(points)
    pts = [np.array(p, dtype=float) for p in points]
    out = [pts[0].tolist()]
    carry = 0.0
    for i in range(1, len(pts)):
        seg = pts[i] - pts[i-1]
        seg_len = float(np.linalg.norm(seg))
        if seg_len < 1e-6:
            continue
        direction = seg / seg_len
        d = carry
        while d < seg_len:
            out.append((pts[i-1] + direction * d).tolist())
            d += step
        carry = d - seg_len
    if np.linalg.norm(np.array(out[-1]) - pts[-1]) > 1e-3:
        out.append(pts[-1].tolist())
    return out


def poly_length(pts: List) -> float:
    arr = np.array(pts, dtype=float)
    return float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1)))


def compute_ndtw(pred: List, ref: List) -> float:
    d, _ = fastdtw(pred, ref, dist=euclidean)
    return float(np.exp(-d / (len(ref) * SUCCESS_DIST)))


def find_path_points(pf, start, end) -> Optional[List]:
    sp = ShortestPath()
    sp.requested_start = mn.Vector3(float(start[0]), float(start[1]), float(start[2]))
    sp.requested_end   = mn.Vector3(float(end[0]),   float(end[1]),   float(end[2]))
    if pf.find_path(sp) and sp.geodesic_distance < float("inf"):
        return [p.tolist() if hasattr(p, "tolist") else list(p) for p in sp.points]
    return None


def build_two_leg_path(pf, start, via, goal):
    leg1 = find_path_points(pf, start, via)
    leg2 = find_path_points(pf, via,   goal)
    if leg1 is None or leg2 is None:
        return None
    return leg1 + leg2[1:]


def is_duplicate(new_rs: List, accepted: List, threshold: float) -> bool:
    for acc in accepted:
        d, _ = fastdtw(new_rs, acc, dist=euclidean)
        if d < threshold:
            return True
    return False


def generate_references(pf, start, goal) -> Tuple[List, int]:
    accepted_rs: List = []
    direct = find_path_points(pf, start, goal)
    if direct:
        accepted_rs.append(resample_path(direct))
    base_len = poly_length(direct) if direct else 5.0
    dedup_thr = base_len * DEDUP_RATIO * len(resample_path([[0,0,0],[base_len,0,0]]))

    pf.seed(VIA_SEED)
    acc, sampled = 0, 0
    while acc < K_VIA and sampled < VIA_OVERSAMPLE:
        via = pf.get_random_navigable_point().tolist()
        sampled += 1
        combined = build_two_leg_path(pf, start, via, goal)
        if combined is None:
            continue
        crs = resample_path(combined)
        if len(crs) < 3:
            continue
        if is_duplicate(crs, accepted_rs, dedup_thr):
            continue
        accepted_rs.append(crs)
        acc += 1
    return accepted_rs, len(accepted_rs)


def make_sim(scene_abs: str) -> habitat_sim.Simulator:
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id           = scene_abs
    backend.create_renderer    = False
    backend.load_semantic_mesh = False
    backend.enable_physics     = True
    backend.gpu_device_id      = 0
    backend.random_seed        = VIA_SEED
    return habitat_sim.Simulator(habitat_sim.Configuration(backend, [habitat_sim.AgentConfiguration()]))


def ep_metrics(ep_id_str, meta, ep_ds,
               base_obs_r, base_clean_r, oap_obs_r, oap_clean_r,
               refs_rs):
    """Compute all scalar metrics for one episode given result dicts."""
    geo_obs  = float(meta["geo_obs"])
    geo_base = float(meta["geo_base"])

    def arm_metrics(r, geo_d0):
        if r is None:
            return dict(SR=float("nan"), PL=float("nan"), dSPL=float("nan"),
                        nDTW=float("nan"), GP=float("nan"), traj=None)
        sr   = float(r.get("success",      0.0))
        pl   = float(r.get("path_length",  0.0))
        ndtw = float(r.get("ndtw",         0.0))
        dtg  = float(r.get("distance_to_goal", geo_d0))
        dspl = sr * geo_d0 / max(geo_d0, pl) if pl > 0 else 0.0
        gp   = max(0.0, (geo_d0 - dtg) / geo_d0) if geo_d0 > 0 else 0.0
        traj = r.get("trajectory", [ep_ds["start_position"]])
        return dict(SR=sr, PL=pl, dSPL=dspl, nDTW=ndtw, GP=gp, traj=traj)

    b   = arm_metrics(base_obs_r,   geo_obs)
    bc  = arm_metrics(base_clean_r, geo_base)
    o   = arm_metrics(oap_obs_r,    geo_obs)
    oc  = arm_metrics(oap_clean_r,  geo_base)

    def mp(traj):
        if traj is None or not refs_rs:
            return float("nan")
        rs = resample_path(traj)
        return max(compute_ndtw(rs, r) for r in refs_rs)

    return {
        "episode_id":       int(ep_id_str),
        "geo_obs":          round(geo_obs, 3),
        "geo_base":         round(geo_base, 3),
        "detour_ratio":     round(float(meta["detour_ratio"]), 3),
        "n_refs_geo":       len(refs_rs),
        # Base
        "Base_SR":          b["SR"],
        "Base_PL":          round(b["PL"], 3),
        "Base_dSPL":        round(b["dSPL"], 4),
        "Base_nDTW":        round(b["nDTW"], 4),
        "Base_mp_ndtw_geo": round(mp(b["traj"]), 4),
        "Base_GP":          round(b["GP"], 4),
        "Base_GP_clean":    round(bc["GP"], 4),
        "Base_PD":          round(bc["GP"] - b["GP"], 4),
        # OAP
        "OAP_SR":           o["SR"],
        "OAP_PL":           round(o["PL"], 3),
        "OAP_dSPL":         round(o["dSPL"], 4),
        "OAP_nDTW":         round(o["nDTW"], 4),
        "OAP_mp_ndtw_geo":  round(mp(o["traj"]), 4),
        "OAP_GP":           round(o["GP"], 4),
        "OAP_GP_clean":     round(oc["GP"], 4),
        "OAP_PD":           round(oc["GP"] - o["GP"], 4),
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    # ── load dataset ──────────────────────────────────────────────────────────
    print("Loading val_unseen dataset...")
    with gzip.open(DATASET_GZ) as f:
        val = json.load(f)
    epmap = {str(e["episode_id"]): e for e in val["episodes"]}

    # ── load manifests ────────────────────────────────────────────────────────
    with open(MANIFEST_500) as f:
        sub500 = json.load(f)
    with open(MANIFEST_91) as f:
        obs91_list = json.load(f)
    obs91_ids = {e["episode_id"] for e in obs91_list}

    meta500 = {e["episode_id"]: e for e in sub500}
    new_409 = [e for e in sub500 if e["episode_id"] not in obs91_ids]
    print(f"Total 500: {len(sub500)}  (91 reused, {len(new_409)} new)")

    # ── load result files ─────────────────────────────────────────────────────
    print("Loading traj result files...")
    base_obs_res   = json.load(open(BASE_OBS))
    base_clean_res = json.load(open(BASE_CLEAN))
    oap_obs_res    = json.load(open(OAP_OBS))
    oap_clean_res  = json.load(open(OAP_CLEAN))

    # ── load 91 existing metrics ──────────────────────────────────────────────
    print("Loading existing 91-ep metrics...")
    with open(METRICS_91) as f:
        m91 = json.load(f)
    existing = {str(v["episode_id"]): v for v in m91["episodes"].values()}
    print(f"  Loaded {len(existing)} existing records")

    # ── resume progress ───────────────────────────────────────────────────────
    computed = {}
    if os.path.exists(PROGRESS_LOG):
        with open(PROGRESS_LOG) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    computed[str(rec["episode_id"])] = rec
        print(f"Resuming: {len(computed)} new episodes already done")

    progress_f = open(PROGRESS_LOG, "a")

    # ── group new 409 by scene ────────────────────────────────────────────────
    by_scene = defaultdict(list)
    for e in new_409:
        by_scene[e["scene_id"]].append(e)

    total_new = len(new_409)
    done_count = len(computed)

    # ── process scene-by-scene ────────────────────────────────────────────────
    for scene_rel, ep_list in sorted(by_scene.items(), key=lambda x: -len(x[1])):
        scene_abs = os.path.join(SCENE_DIR, scene_rel)
        scene_name = scene_rel.split("/")[-1]
        to_do = [e for e in ep_list if str(e["episode_id"]) not in computed]
        if not to_do:
            print(f"Scene {scene_name}: all {len(ep_list)} eps cached, skip")
            continue

        print(f"\n{'='*60}")
        print(f"Scene {scene_name}: {len(ep_list)} total, {len(to_do)} to compute")
        print(f"Loading sim: {scene_abs}")
        sim = make_sim(scene_abs)
        injector = ObstacleInjector(sim, recompute_navmesh=True)

        for ep_meta in sorted(to_do, key=lambda x: x["episode_id"]):
            eid = str(ep_meta["episode_id"])
            done_count += 1
            t_ep = time.time()
            print(f"\n  [{done_count:3d}/{total_new}] ep {eid}")

            ep_ds = epmap.get(eid)
            if ep_ds is None:
                print(f"  WARNING: ep {eid} not in dataset, skip")
                continue

            start = ep_ds["start_position"]
            goal  = ep_ds["goals"][0]["position"]

            # place obstacle + recompute navmesh
            injector.clear_all()
            spec = ObstacleSpec(
                position=(float(ep_meta["x"]), float(ep_meta["y"]), float(ep_meta["z"])),
                size=BOX_SIZE,
                yaw_deg=float(ep_meta["yaw_deg"]),
            )
            injector.add_obstacle(spec)

            refs_rs, n_refs = generate_references(sim.pathfinder, start, goal)
            print(f"  {n_refs} refs")

            b_obs_r   = base_obs_res.get(eid)
            b_cln_r   = base_clean_res.get(eid)
            o_obs_r   = oap_obs_res.get(eid)
            o_cln_r   = oap_clean_res.get(eid)

            rec = ep_metrics(eid, ep_meta, ep_ds,
                             b_obs_r, b_cln_r, o_obs_r, o_cln_r,
                             refs_rs)

            print(f"  Base SR={rec['Base_SR']:.0f} dSPL={rec['Base_dSPL']:.4f} "
                  f"nDTW={rec['Base_nDTW']:.4f} mpGeo={rec['Base_mp_ndtw_geo']:.4f} "
                  f"GP={rec['Base_GP']:.4f} PD={rec['Base_PD']:.4f}")
            print(f"  OAP  SR={rec['OAP_SR']:.0f} dSPL={rec['OAP_dSPL']:.4f} "
                  f"nDTW={rec['OAP_nDTW']:.4f} mpGeo={rec['OAP_mp_ndtw_geo']:.4f} "
                  f"GP={rec['OAP_GP']:.4f} PD={rec['OAP_PD']:.4f}")
            print(f"  [{time.time()-t_ep:.1f}s]")

            computed[eid] = rec
            progress_f.write(json.dumps(rec) + "\n")
            progress_f.flush()

        sim.close()
        print(f"Scene {scene_name} done.")

    progress_f.close()
    print(f"\nAll {len(computed)} new episodes computed.")

    # ── merge: 91 existing + 409 new ─────────────────────────────────────────
    print("Merging with 91 existing metrics...")
    all_records = {}

    # 91 existing: adapt field names (metrics_91ep uses same schema)
    for eid_str, rec in existing.items():
        ep_id = int(eid_str)
        if ep_id not in meta500:
            continue  # not in our 500 subset
        meta = meta500[ep_id]
        r = dict(rec)
        r["geo_base"] = round(float(meta.get("geo_base", float("nan"))), 3)
        all_records[eid_str] = r

    # 409 new
    for eid_str, rec in computed.items():
        all_records[eid_str] = rec

    print(f"Total merged: {len(all_records)} episodes (expected 500)")

    # ── aggregate statistics ──────────────────────────────────────────────────
    def mean(vals):
        v = [x for x in vals if not (isinstance(x, float) and x != x)]
        return sum(v) / len(v) if v else float("nan")

    keys = list(all_records.values())
    print("\n=== 500-ep Aggregate Results ===")
    for prefix, label in [("Base", "Base (no prefix)"), ("OAP", "OAP (prefix)")]:
        sr   = mean([r[f"{prefix}_SR"]           for r in keys])
        dspl = mean([r[f"{prefix}_dSPL"]         for r in keys])
        ndtw = mean([r[f"{prefix}_nDTW"]         for r in keys])
        mp   = mean([r[f"{prefix}_mp_ndtw_geo"]  for r in keys])
        gp   = mean([r[f"{prefix}_GP"]           for r in keys])
        gpc  = mean([r[f"{prefix}_GP_clean"]     for r in keys])
        pd   = mean([r[f"{prefix}_PD"]           for r in keys])
        print(f"  {label}:")
        print(f"    SR={sr:.4f}  dSPL={dspl:.4f}  nDTW={ndtw:.4f}  mp_ndtw_geo={mp:.4f}")
        print(f"    GP={gp:.4f}  GP_clean={gpc:.4f}  PD={pd:.4f}")

    print(f"\n  Base vs OAP diffs:")
    print(f"    ΔSR={mean([r['OAP_SR']-r['Base_SR'] for r in keys]):+.4f}")
    print(f"    ΔdSPL={mean([r['OAP_dSPL']-r['Base_dSPL'] for r in keys]):+.4f}")
    print(f"    Δmp_ndtw_geo={mean([r['OAP_mp_ndtw_geo']-r['Base_mp_ndtw_geo'] for r in keys]):+.4f}")
    print(f"    ΔGP={mean([r['OAP_GP']-r['Base_GP'] for r in keys]):+.4f}")

    # ── save ─────────────────────────────────────────────────────────────────
    out = {"n_episodes": len(all_records), "episodes": all_records}
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_JSON}")

    fields = ["episode_id","geo_obs","geo_base","detour_ratio","n_refs_geo",
              "Base_SR","Base_PL","Base_dSPL","Base_nDTW","Base_mp_ndtw_geo",
              "Base_GP","Base_GP_clean","Base_PD",
              "OAP_SR","OAP_PL","OAP_dSPL","OAP_nDTW","OAP_mp_ndtw_geo",
              "OAP_GP","OAP_GP_clean","OAP_PD"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for rec in sorted(all_records.values(), key=lambda x: x["episode_id"]):
            w.writerow(rec)
    print(f"Saved: {OUT_CSV}")
    print(f"\nTotal elapsed: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
