"""
Habitat-sim 장애물 주입 유틸리티.

씬에 정적(static) 박스 장애물을 추가·제거하고 NavMesh를 재계산한다.
에피소드 경계(reset)에서 호출해 VLN 평가의 robustness 실험에 활용한다.
"""

import logging
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

import habitat_sim
import magnum as mn
import numpy as np
from habitat_sim.nav import NavMeshSettings, ShortestPath
from habitat_sim.physics import MotionType


# ─── 장애물 명세 ──────────────────────────────────────────────────────────────

@dataclass
class ObstacleSpec:
    """씬에 삽입할 박스 장애물 하나의 설정."""

    position: Tuple[float, float, float]   # (x, y, z) 월드 좌표 [m]
    size: Tuple[float, float, float] = (0.4, 1.2, 0.4)  # (width, height, depth) [m]
    yaw_deg: float = 0.0                   # Y축 회전 [도]


@dataclass
class RealObjectSpec:
    """씬에 삽입할 실제 3D 오브젝트(YCB 등) 하나의 설정."""

    position: Tuple[float, float, float]   # (x, y, z) 월드 좌표 [m]
    handle: str                            # .object_config.json 절대 경로
    yaw_deg: float = 0.0                   # Y축 회전 [도]

    @property
    def name(self) -> str:
        """파일명에서 오브젝트 이름만 추출."""
        import os
        return os.path.basename(self.handle).replace(".object_config.json", "")


# YCB 장애물용 기본 오브젝트 이름 (box-like: 복도 차단에 효과적)
YCB_BLOCKING_NAMES: Tuple[str, ...] = (
    "003_cracker_box",
    "004_sugar_box",
    "006_mustard_bottle",
    "010_potted_meat_can",
    "019_pitcher_base",
    "021_bleach_cleanser",
    "025_mug",
    "036_wood_block",
)

# Replica CAD 가구 중 내비게이션 차단에 충분한 크기의 오브젝트
# 실측 bbox: sofa(2.14×0.80×0.96m), chair_01(0.80×0.76×0.71m),
#            table_01(1.40×0.74×0.85m) — scale_factor=1.0으로 사용
REPLICA_FURNITURE_NAMES: Tuple[str, ...] = (
    "frl_apartment_sofa",
    "frl_apartment_chair_01",
    "frl_apartment_chair_04",
    "frl_apartment_chair_05",
    "frl_apartment_table_01",
    "frl_apartment_table_02",
    "frl_apartment_table_03",
    "frl_apartment_table_04",
    "frl_apartment_cabinet",
    "frl_apartment_refrigerator",
    "frl_apartment_tvstand",
    "frl_apartment_stool_02",
)

REPLICA_CONFIG_DIR_DEFAULT = os.environ.get(
    "REPLICA_CONFIG_DIR",
    "../../habitat-lab/data/versioned_data/replica_cad_dataset/configs/objects",
)

# ─── 메인 클래스 ──────────────────────────────────────────────────────────────

class ObstacleInjector:
    """
    Habitat-sim Simulator 위에서 동작하는 장애물 관리자.

    사용 예시
    ----------
    injector = ObstacleInjector(sim)
    obj_id = injector.add_obstacle(ObstacleSpec(position=(1.0, 0.5, 2.0)))
    ...
    injector.clear_all()
    """

    _OBJ_TEMPLATE_NAME  = "_navila_obstacle_box"
    _PRIM_ASSET_HANDLE  = "cubeSolid"  # habitat_sim 기본 등록 primitive

    def __init__(self, sim, recompute_navmesh: bool = True, navmesh_settings: Optional[NavMeshSettings] = None):
        """
        Parameters
        ----------
        sim                : habitat_sim.Simulator
        recompute_navmesh  : True면 장애물 추가·제거 시 NavMesh 자동 재계산
        navmesh_settings   : None이면 기본값 사용
        """
        self._sim = sim
        self._recompute = recompute_navmesh
        self._navmesh_settings = navmesh_settings or self._default_navmesh_settings()
        self._injected: List[int] = []          # 추가된 오브젝트 id 목록
        self._template_registered = False
        self._episode_stats: List[Dict] = []    # 에피소드별 배치 통계
        self._ycb_handles: List[str] = []       # load_ycb_templates() 후 사용 가능

    # ── 공개 API ──────────────────────────────────────────────────────────────

    def add_obstacle(self, spec: ObstacleSpec) -> int:
        """장애물을 씬에 추가하고 오브젝트 id를 반환한다."""
        self._ensure_template(spec.size)

        obj_id = self._sim.add_object_by_handle(self._OBJ_TEMPLATE_NAME)

        pos = mn.Vector3(spec.position[0], spec.position[1], spec.position[2])
        self._sim.set_translation(pos, obj_id)

        if spec.yaw_deg != 0.0:
            rad = np.deg2rad(spec.yaw_deg)
            quat = mn.Quaternion.rotation(mn.Rad(rad), mn.Vector3(0, 1, 0))
            self._sim.set_rotation(quat, obj_id)

        self._sim.set_object_motion_type(MotionType.STATIC, obj_id)
        self._sim.set_object_is_collidable(True, obj_id)

        self._injected.append(obj_id)

        if self._recompute:
            self._recompute_navmesh()

        return obj_id

    def add_obstacles(self, specs: List[ObstacleSpec], batch_navmesh: bool = True) -> List[int]:
        """여러 장애물을 한 번에 추가한다. batch_navmesh=True면 마지막에 한 번만 NavMesh 재계산."""
        prev = self._recompute
        self._recompute = False
        ids = [self.add_obstacle(s) for s in specs]
        self._recompute = prev

        if batch_navmesh and self._recompute:
            self._recompute_navmesh()

        return ids

    def add_obstacles_along_path(
        self,
        waypoints: List[List[float]],
        n_obstacles: int = 3,
        lateral_spread: float = 0.8,
        size: Tuple[float, float, float] = (0.4, 1.2, 0.4),
        rng: Optional[random.Random] = None,
        episode_id: Optional[int] = None,
        base_seed: int = 42,
    ) -> List[int]:
        """
        경로(waypoints) 주변에 장애물을 랜덤하게 배치한다.

        적용 조건
        ---------
        1. 벽 거리 검사: 장애물 중심~벽 거리 < 장애물 크기(XZ) 이면 배치 안 함
        2. 경로 차단 검사: 배치 후 출발지→목적지 NavMesh 경로 없으면 제거
        3. 재현 가능 시드: episode_id 지정 시 random.Random(base_seed + episode_id) 사용

        Parameters
        ----------
        waypoints      : [[x,y,z], ...] 형식의 경로 점 목록
        n_obstacles    : 삽입할 장애물 수
        lateral_spread : 경로에서 좌우로 흩뿌릴 최대 반경 [m]
        size           : 각 장애물의 (width, height, depth)
        rng            : seeded random.Random 인스턴스 (episode_id 미지정 시 사용)
        episode_id     : 지정하면 random.Random(base_seed + episode_id)로 내부 시딩
        base_seed      : episode_id 기반 시딩의 기본값 (기본 42)
        """
        if len(waypoints) < 2 or n_obstacles <= 0:
            return []

        # 조건 3: episode_id 기반 재현 가능한 시드
        if episode_id is not None:
            rng = random.Random(base_seed + episode_id)
        else:
            rng = rng or random.Random()

        start = np.array(waypoints[0], dtype=float)
        goal  = np.array(waypoints[-1], dtype=float)
        placed_ids: List[int] = []

        ep_stat: Dict = {
            "episode_id":        episode_id,
            "n_requested":       n_obstacles,
            "n_attempted":       0,
            "n_wall_rejected":   0,
            "n_detour_rejected": 0,
            "n_zero_seg":        0,  # 길이 0인 세그먼트 스킵
            "n_placed":          0,
        }

        for i in range(n_obstacles):
            # 경로 세그먼트 중 랜덤 선택
            idx = rng.randint(0, len(waypoints) - 2)
            p0 = np.array(waypoints[idx], dtype=float)
            p1 = np.array(waypoints[idx + 1], dtype=float)

            # 진행 방향 수직벡터 계산 (XZ 평면)
            fwd = p1 - p0
            fwd[1] = 0.0
            norm = np.linalg.norm(fwd)
            if norm < 1e-6:
                ep_stat["n_zero_seg"] += 1
                logger.debug(
                    "[ObstacleInjector] ep=%s slot=%d: zero-length segment (seg_idx=%d) — skipped",
                    episode_id, i, idx,
                )
                continue
            fwd /= norm
            right = np.array([-fwd[2], 0.0, fwd[0]])

            # 세그먼트 위에서 랜덤 위치 + 좌우 흩뿌리기
            t       = rng.random()
            center  = p0 + t * (p1 - p0)
            lateral = rng.uniform(-lateral_spread, lateral_spread)
            pos     = center + lateral * right
            yaw     = rng.uniform(0, 360)

            # Y: 바닥에 붙도록 절반 높이만큼 위로
            pos[1] = p0[1] + size[1] / 2.0

            ep_stat["n_attempted"] += 1

            # 조건 1: 벽 거리 검사 (물리 오브젝트 배치 전에 빠르게 reject)
            wall_dist = self._distance_to_wall(pos, size)
            min_dist  = max(size[0], size[2])
            if wall_dist < min_dist:
                ep_stat["n_wall_rejected"] += 1
                logger.debug(
                    "[ObstacleInjector] ep=%s slot=%d: WALL_REJECT pos=(%.2f,%.2f,%.2f) "
                    "wall_dist=%.3f < threshold=%.3f",
                    episode_id, i,
                    pos[0], pos[1], pos[2],
                    wall_dist, min_dist,
                )
                continue

            spec = ObstacleSpec(
                position=(float(pos[0]), float(pos[1]), float(pos[2])),
                size=size,
                yaw_deg=yaw,
            )

            # 조건 2: 배치 후 NavMesh 우회 경로 확인
            obj_id = self._place_if_detour_exists(spec, start, goal)
            if obj_id >= 0:
                placed_ids.append(obj_id)
                ep_stat["n_placed"] += 1
                logger.debug(
                    "[ObstacleInjector] ep=%s slot=%d: PLACED obj_id=%d pos=(%.2f,%.2f,%.2f)",
                    episode_id, i, obj_id,
                    pos[0], pos[1], pos[2],
                )
            else:
                ep_stat["n_detour_rejected"] += 1
                logger.debug(
                    "[ObstacleInjector] ep=%s slot=%d: DETOUR_REJECT pos=(%.2f,%.2f,%.2f) "
                    "(no navigable bypass after placement)",
                    episode_id, i,
                    pos[0], pos[1], pos[2],
                )

        self._episode_stats.append(ep_stat)
        success_rate = ep_stat["n_placed"] / ep_stat["n_requested"] if ep_stat["n_requested"] > 0 else 0.0
        logger.info(
            "[ObstacleInjector] ep=%s | placed=%d/%d (%.0f%%) "
            "wall_rej=%d detour_rej=%d zero_seg=%d",
            episode_id,
            ep_stat["n_placed"], ep_stat["n_requested"],
            success_rate * 100,
            ep_stat["n_wall_rejected"],
            ep_stat["n_detour_rejected"],
            ep_stat["n_zero_seg"],
        )

        return placed_ids

    def add_obstacle_at_path_ratio(
        self,
        waypoints: List[List[float]],
        ratio: float = 0.4,
        size: Tuple[float, float, float] = (0.4, 1.2, 0.4),
        yaw_deg: float = 0.0,
    ) -> int:
        """
        경로 위 ratio 위치에 장애물을 직접 배치한다 (lateral spread 없음).
        배치 후 NavMesh로 우회 경로 존재 여부를 확인하고,
        우회 경로가 없으면 배치하지 않고 -1을 반환한다.

        Parameters
        ----------
        waypoints : [[x,y,z], ...] 형식의 경로 점 목록
        ratio     : 경로상 위치 (0.0=시작, 1.0=목표). 기본 0.4
        size      : (width, height, depth) [m]
        yaw_deg   : Y축 회전 [도]

        Returns
        -------
        오브젝트 id (배치 실패 시 -1)
        """
        if len(waypoints) < 2:
            raise ValueError("waypoints must have at least 2 points")

        idx = int(ratio * (len(waypoints) - 1))
        idx = max(1, min(idx, len(waypoints) - 2))
        pos = np.array(waypoints[idx], dtype=float)
        pos[1] += size[1] / 2.0  # 바닥에 붙이기

        spec = ObstacleSpec(
            position=(float(pos[0]), float(pos[1]), float(pos[2])),
            size=size,
            yaw_deg=yaw_deg,
        )
        start = np.array(waypoints[0], dtype=float)
        goal  = np.array(waypoints[-1], dtype=float)
        return self._place_if_detour_exists(spec, start, goal)

    def remove_obstacle(self, obj_id: int) -> None:
        """특정 장애물을 제거한다."""
        if obj_id in self._injected:
            self._sim.remove_object(obj_id)
            self._injected.remove(obj_id)
            if self._recompute:
                self._recompute_navmesh()

    def clear_all(self) -> None:
        """삽입된 모든 장애물을 제거하고 NavMesh를 복원한다."""
        for obj_id in list(self._injected):
            self._sim.remove_object(obj_id)
        self._injected.clear()
        if self._recompute:
            self._recompute_navmesh()

    @property
    def injected_ids(self) -> List[int]:
        return list(self._injected)

    def get_placement_stats(self, warn_threshold: float = 0.5) -> Dict:
        """에피소드별 장애물 배치 통계를 반환한다.

        전체 성공률이 warn_threshold 미만이면 완화 방안을 WARNING 레벨로 제안한다.

        Returns
        -------
        dict with keys:
          n_episodes, total_requested, total_placed, total_wall_rejected,
          total_detour_rejected, success_rate, per_episode
        """
        if not self._episode_stats:
            logger.info("[ObstacleInjector] 아직 기록된 에피소드 통계가 없습니다.")
            return {}

        total_req    = sum(s["n_requested"]       for s in self._episode_stats)
        total_placed = sum(s["n_placed"]           for s in self._episode_stats)
        total_wall   = sum(s["n_wall_rejected"]    for s in self._episode_stats)
        total_detour = sum(s["n_detour_rejected"]  for s in self._episode_stats)
        total_zero   = sum(s["n_zero_seg"]         for s in self._episode_stats)
        success_rate = total_placed / total_req if total_req > 0 else 0.0

        summary = {
            "n_episodes":            len(self._episode_stats),
            "total_requested":       total_req,
            "total_placed":          total_placed,
            "total_wall_rejected":   total_wall,
            "total_detour_rejected": total_detour,
            "total_zero_seg":        total_zero,
            "success_rate":          success_rate,
            "per_episode":           list(self._episode_stats),
        }

        logger.info(
            "[ObstacleInjector] ── 전체 통계 (%d 에피소드) ──\n"
            "  요청: %d  배치 성공: %d  성공률: %.1f%%\n"
            "  벽 거리 거부: %d  경로 차단 거부: %d  제로세그 스킵: %d",
            summary["n_episodes"],
            total_req, total_placed, success_rate * 100,
            total_wall, total_detour, total_zero,
        )

        if success_rate < warn_threshold:
            self._suggest_relaxation(success_rate, total_wall, total_detour, total_req)

        return summary

    def reset_stats(self) -> None:
        """누적된 에피소드 통계를 초기화한다."""
        self._episode_stats.clear()

    # ── YCB 실제 오브젝트 API ─────────────────────────────────────────────────

    def load_ycb_templates(
        self,
        ycb_config_dir: str,
        scale_factor: float = 4.0,
        blocking_only: bool = True,
    ) -> List[str]:
        """YCB 오브젝트 템플릿을 등록하고 핸들 목록을 반환한다.

        Parameters
        ----------
        ycb_config_dir  : YCB configs 디렉터리 절대 경로
                          (예: .../ycb/configs)
        scale_factor    : 전체 스케일 배수 (기본 4.0 → 복도 차단에 충분한 크기)
        blocking_only   : True면 YCB_BLOCKING_NAMES 서브셋만 등록
        """
        import os
        mgr = self._sim.get_object_template_manager()
        mgr.load_configs(ycb_config_dir)
        all_handles = [
            h for h in mgr.get_file_template_handles()
            if ycb_config_dir in h
        ]

        selected: List[str] = []
        for h in all_handles:
            name = os.path.basename(h).replace(".object_config.json", "")
            if blocking_only and not any(name == b for b in YCB_BLOCKING_NAMES):
                continue
            t = mgr.get_template_by_handle(h)
            t.scale = mn.Vector3(scale_factor, scale_factor, scale_factor)
            t.mass = 0.0
            mgr.register_template(t, h)
            selected.append(h)

        self._ycb_handles = selected
        logger.info(
            "[ObstacleInjector] YCB 템플릿 등록 완료: %d개 (scale=%.1f, blocking_only=%s)",
            len(selected), scale_factor, blocking_only,
        )
        return selected

    def load_replica_templates(
        self,
        replica_config_dir: str = REPLICA_CONFIG_DIR_DEFAULT,
        scale_factor: float = 1.0,
        furniture_only: bool = True,
    ) -> List[str]:
        """Replica CAD 가구 템플릿을 등록하고 핸들 목록을 반환한다.

        Replica 오브젝트는 이미 실물 크기이므로 scale_factor 기본값은 1.0.
        render_asset_handle을 절대경로로 명시해 gray-box 렌더링을 방지한다
        (config JSON의 상대경로 ../../objects/...가 버전에 따라 미해석될 수 있음).

        Parameters
        ----------
        replica_config_dir : Replica configs/objects 디렉터리 절대 경로
        scale_factor       : 전체 스케일 배수 (기본 1.0)
        furniture_only     : True면 REPLICA_FURNITURE_NAMES 서브셋만 등록
        """
        import os
        mgr = self._sim.get_object_template_manager()
        mgr.load_configs(replica_config_dir)
        all_handles = [
            h for h in mgr.get_file_template_handles()
            if replica_config_dir in h
        ]

        # config JSON의 render_asset 상대경로(../../objects/XXX.glb)를 절대경로로 변환
        # configs/objects/ → ../../ → replica_cad_dataset/objects/
        objects_dir = os.path.normpath(
            os.path.join(replica_config_dir, "..", "..", "objects")
        )

        selected: List[str] = []
        for h in all_handles:
            name = os.path.basename(h).replace(".object_config.json", "")
            if furniture_only and name not in REPLICA_FURNITURE_NAMES:
                continue
            t = mgr.get_template_by_handle(h)

            # render mesh를 절대경로로 명시 — collision-only mesh로 렌더되는 것을 방지
            render_glb = os.path.join(objects_dir, f"{name}.glb")
            if os.path.isfile(render_glb):
                t.render_asset_handle = render_glb
            else:
                logger.warning(
                    "[ObstacleInjector] render mesh not found: %s (will use config default)",
                    render_glb,
                )

            if scale_factor != 1.0:
                t.scale = mn.Vector3(scale_factor, scale_factor, scale_factor)
            t.mass = 0.0   # 물리 엔진이 움직이지 않도록 static
            mgr.register_template(t, h)
            selected.append(h)

        self._ycb_handles = selected   # 기존 파이프라인과 동일한 필드에 저장
        logger.info(
            "[ObstacleInjector] Replica 템플릿 등록 완료: %d개 (scale=%.1f, furniture_only=%s)",
            len(selected), scale_factor, furniture_only,
        )
        return selected

    def add_real_object(self, spec: RealObjectSpec) -> int:
        """실제 YCB 오브젝트를 씬에 추가하고 오브젝트 id를 반환한다.

        load_ycb_templates() 또는 사전에 등록된 핸들이어야 한다.
        """
        obj_id = self._sim.add_object_by_handle(spec.handle)
        if obj_id < 0:
            logger.warning(
                "[ObstacleInjector] add_object_by_handle('%s') 실패 (id=%d)",
                spec.name, obj_id,
            )
            return obj_id

        pos = mn.Vector3(spec.position[0], spec.position[1], spec.position[2])
        self._sim.set_translation(pos, obj_id)

        if spec.yaw_deg != 0.0:
            rad = np.deg2rad(spec.yaw_deg)
            quat = mn.Quaternion.rotation(mn.Rad(rad), mn.Vector3(0, 1, 0))
            self._sim.set_rotation(quat, obj_id)

        self._sim.set_object_motion_type(MotionType.STATIC, obj_id)
        self._sim.set_object_is_collidable(True, obj_id)
        self._injected.append(obj_id)

        if self._recompute:
            self._recompute_navmesh()

        return obj_id

    def add_real_objects_along_path(
        self,
        waypoints: List[List[float]],
        n_obstacles: int = 3,
        lateral_spread: float = 0.8,
        scale_factor: float = 4.0,
        ycb_config_dir: Optional[str] = None,
        obj_config_dir: Optional[str] = None,   # ycb_config_dir 별칭 (Replica 등 범용)
        blocking_only: bool = True,
        rng: Optional[random.Random] = None,
        episode_id: Optional[int] = None,
        base_seed: int = 42,
        y_offset: Optional[float] = None,       # 바닥 기준 오브젝트 중심 높이 [m]
        obj_half_extent: Optional[float] = None, # 벽 거리 검사용 오브젝트 반경 [m]
    ) -> List[int]:
        """경로(waypoints) 주변에 실제 3D 오브젝트를 랜덤하게 배치한다.

        YCB 및 Replica 모두 지원. 벽 거리·경로 차단 검사는 box 버전과 동일.

        Parameters
        ----------
        waypoints        : [[x,y,z], ...] 형식의 경로 점 목록
        n_obstacles      : 삽입할 오브젝트 수
        lateral_spread   : 경로에서 좌우로 흩뿌릴 최대 반경 [m]
        scale_factor     : 오브젝트 스케일 배수 (YCB=4.0, Replica=1.0)
        ycb_config_dir   : (YCB) configs 디렉터리 — load_ycb_templates() lazy 호출
        obj_config_dir   : ycb_config_dir 별칭. Replica 등 범용 사용 시 지정
        blocking_only    : True면 YCB_BLOCKING_NAMES만 필터
        rng              : seeded random.Random 인스턴스
        episode_id       : 지정하면 random.Random(base_seed + episode_id)로 시딩
        base_seed        : episode_id 기반 시딩 기본값
        y_offset         : 오브젝트 중심 Y = floor_y + y_offset [m].
                           None → scale_factor * 0.15 (YCB 기본값).
                           Replica 가구에는 0.4 권장.
        obj_half_extent  : 벽 거리 검사용 오브젝트 XZ 반경 [m].
                           None → scale_factor * 0.08 (YCB 기본값).
                           Replica 가구에는 0.45 권장.

        Returns
        -------
        배치된 오브젝트 id 목록
        """
        if len(waypoints) < 2 or n_obstacles <= 0:
            return []

        # obj_config_dir이 ycb_config_dir의 범용 별칭
        _config_dir = obj_config_dir or ycb_config_dir

        # 오브젝트 템플릿 lazy 로드 (핸들이 없을 때만)
        if _config_dir and not self._ycb_handles:
            self.load_ycb_templates(_config_dir, scale_factor=scale_factor, blocking_only=blocking_only)
        if not self._ycb_handles:
            logger.error(
                "[ObstacleInjector] 오브젝트 핸들 없음. "
                "load_ycb_templates()/load_replica_templates() 또는 "
                "ycb_config_dir/obj_config_dir를 지정하세요."
            )
            return []

        if episode_id is not None:
            rng = random.Random(base_seed + episode_id)
        else:
            rng = rng or random.Random()

        start = np.array(waypoints[0], dtype=float)
        goal  = np.array(waypoints[-1], dtype=float)
        placed_ids: List[int] = []

        # 벽 거리 검사용 근사 크기
        # YCB: scale_factor * 0.08 (YCB 폭 ~0.08m × scale)
        # Replica: obj_half_extent=0.45 (의자 폭 ~0.8m → 반경 0.4m)
        _half_ext  = obj_half_extent if obj_half_extent is not None else scale_factor * 0.08
        # Y 배치 오프셋: 바닥면에서 오브젝트 중심까지 높이
        # YCB: scale_factor * 0.15  Replica 가구: 0.4m
        _y_offset  = y_offset if y_offset is not None else scale_factor * 0.15

        ep_stat: Dict = {
            "episode_id":        episode_id,
            "mode":              "real_object",
            "n_requested":       n_obstacles,
            "n_attempted":       0,
            "n_wall_rejected":   0,
            "n_detour_rejected": 0,
            "n_zero_seg":        0,
            "n_placed":          0,
        }

        for i in range(n_obstacles):
            idx = rng.randint(0, len(waypoints) - 2)
            p0 = np.array(waypoints[idx], dtype=float)
            p1 = np.array(waypoints[idx + 1], dtype=float)

            fwd = p1 - p0
            fwd[1] = 0.0
            norm = np.linalg.norm(fwd)
            if norm < 1e-6:
                ep_stat["n_zero_seg"] += 1
                logger.debug(
                    "[ObstacleInjector] ep=%s real slot=%d: zero-length segment — skipped",
                    episode_id, i,
                )
                continue
            fwd /= norm
            right = np.array([-fwd[2], 0.0, fwd[0]])

            t_val   = rng.random()
            center  = p0 + t_val * (p1 - p0)
            lateral = rng.uniform(-lateral_spread, lateral_spread)
            pos     = center + lateral * right
            yaw     = rng.uniform(0, 360)

            # Y: 웨이포인트 세그먼트의 y를 바닥 기준으로 사용하고 _y_offset 추가.
            # 웨이포인트의 y는 실제 에피소드 경로 baselevel이므로 NavMesh snap보다 신뢰도 높음.
            # NavMesh snap은 카운터·계단 등 상위 서피스로 잘못 스냅될 수 있음.
            _floor_y = float(center[1])
            pos[1] = _floor_y + _y_offset

            ep_stat["n_attempted"] += 1

            # 벽 거리 검사 (근사 크기 사용)
            approx_size = (_half_ext * 2, _y_offset * 2, _half_ext * 2)
            wall_dist = self._distance_to_wall(pos, approx_size)
            min_dist  = _half_ext
            if wall_dist < min_dist:
                ep_stat["n_wall_rejected"] += 1
                logger.debug(
                    "[ObstacleInjector] ep=%s real slot=%d: WALL_REJECT pos=(%.2f,%.2f,%.2f) "
                    "wall_dist=%.3f < threshold=%.3f",
                    episode_id, i, pos[0], pos[1], pos[2], wall_dist, min_dist,
                )
                continue

            # 랜덤 YCB 핸들 선택
            handle = rng.choice(self._ycb_handles)
            import os
            obj_name = os.path.basename(handle).replace(".object_config.json", "")

            real_spec = RealObjectSpec(
                position=(float(pos[0]), float(pos[1]), float(pos[2])),
                handle=handle,
                yaw_deg=yaw,
            )

            obj_id = self._place_real_object_if_detour_exists(real_spec, start, goal)
            if obj_id >= 0:
                placed_ids.append(obj_id)
                ep_stat["n_placed"] += 1
                logger.debug(
                    "[ObstacleInjector] ep=%s real slot=%d: PLACED %s obj_id=%d pos=(%.2f,%.2f,%.2f)",
                    episode_id, i, obj_name, obj_id, pos[0], pos[1], pos[2],
                )
            else:
                ep_stat["n_detour_rejected"] += 1
                logger.debug(
                    "[ObstacleInjector] ep=%s real slot=%d: DETOUR_REJECT %s pos=(%.2f,%.2f,%.2f)",
                    episode_id, i, obj_name, pos[0], pos[1], pos[2],
                )

        self._episode_stats.append(ep_stat)
        success_rate = ep_stat["n_placed"] / ep_stat["n_requested"] if ep_stat["n_requested"] > 0 else 0.0
        logger.info(
            "[ObstacleInjector] ep=%s [real] | placed=%d/%d (%.0f%%) "
            "wall_rej=%d detour_rej=%d zero_seg=%d",
            episode_id,
            ep_stat["n_placed"], ep_stat["n_requested"],
            success_rate * 100,
            ep_stat["n_wall_rejected"],
            ep_stat["n_detour_rejected"],
            ep_stat["n_zero_seg"],
        )
        return placed_ids

    # ── 내부 메서드 ───────────────────────────────────────────────────────────

    def _suggest_relaxation(
        self,
        success_rate: float,
        n_wall: int,
        n_detour: int,
        n_total: int,
    ) -> None:
        """배치 성공률이 낮을 때 완화 방안을 WARNING으로 출력한다."""
        wall_ratio   = n_wall   / n_total if n_total > 0 else 0.0
        detour_ratio = n_detour / n_total if n_total > 0 else 0.0

        logger.warning(
            "[ObstacleInjector] 배치 성공률 %.1f%% — 완화 방안 제안:",
            success_rate * 100,
        )
        if wall_ratio > 0.2:
            logger.warning(
                "  [벽 거리 거부 %.0f%%] "
                "→ _is_too_close_to_wall threshold를 낮추거나 장애물 크기를 줄이세요.\n"
                "    예) size=(0.4,1.2,0.4) → (0.3,1.2,0.3)  또는\n"
                "        min_dist = max(size[0],size[2]) * 0.7  (완화 계수 적용)\n"
                "    또는 lateral_spread를 줄여 벽 근처 위치를 피하세요.",
                wall_ratio * 100,
            )
        if detour_ratio > 0.2:
            logger.warning(
                "  [경로 차단 거부 %.0f%%] "
                "→ 장애물이 좁은 복도를 완전히 막는 경우입니다.\n"
                "    예) size 줄이기: (0.4,1.2,0.4) → (0.3,1.2,0.3)\n"
                "        lateral_spread 늘리기: 0.8 → 1.2 (경로 중심 회피)\n"
                "        n_obstacles 줄이기: 3 → 2 (누적 차단 방지)",
                detour_ratio * 100,
            )
        if wall_ratio <= 0.2 and detour_ratio <= 0.2:
            logger.warning(
                "  [기타] 세그먼트 길이 부족 또는 waypoints 부족일 수 있습니다.\n"
                "    → waypoints 밀도를 높이거나 경로 길이가 긴 에피소드를 선택하세요."
            )

    def _distance_to_wall(
        self,
        pos: np.ndarray,
        size: Tuple[float, float, float],
    ) -> float:
        """장애물 중심에서 NavMesh 기준 가장 가까운 장애물(벽)까지의 거리를 반환한다.

        pos가 NavMesh 위에 없으면(non-navigable) snap_point로 보정 후 쿼리한다.
        비-navigable 지점에서 distance_to_closest_obstacle는 0을 반환하므로
        snap 없이 쿼리하면 모든 위치가 벽 거부로 처리된다.
        """
        min_dist = max(size[0], size[2])
        max_radius = min_dist * 2 + 0.5
        point = mn.Vector3(float(pos[0]), float(pos[1]), float(pos[2]))
        pf = self._sim.pathfinder

        if not pf.is_navigable(point):
            snapped = pf.snap_point(point)
            if snapped is None:
                return 0.0
            # XZ만 snap하고 y는 원래 높이 유지 (상위 서피스로의 오스냅 방지)
            point = mn.Vector3(float(snapped[0]), float(point[1]), float(snapped[2]))

        return pf.distance_to_closest_obstacle(point, max_search_radius=max_radius)

    def _is_too_close_to_wall(
        self,
        pos: np.ndarray,
        size: Tuple[float, float, float],
    ) -> bool:
        """장애물 중심에서 벽까지 거리가 장애물 크기(XZ 최대 변) 미만이면 True."""
        return self._distance_to_wall(pos, size) < max(size[0], size[2])

    def _place_real_object_if_detour_exists(
        self,
        spec: "RealObjectSpec",
        start: np.ndarray,
        goal: np.ndarray,
    ) -> int:
        """실제 오브젝트를 배치하고 NavMesh 우회 경로 존재 여부를 확인한다."""
        prev_recompute = self._recompute
        self._recompute = False
        obj_id = self.add_real_object(spec)
        if obj_id < 0:
            self._recompute = prev_recompute
            return -1
        self._recompute_navmesh()
        self._recompute = prev_recompute

        if self._detour_exists(start, goal):
            return obj_id

        self._sim.remove_object(obj_id)
        if obj_id in self._injected:
            self._injected.remove(obj_id)
        self._recompute_navmesh()
        return -1

    def _place_if_detour_exists(
        self,
        spec: ObstacleSpec,
        start: np.ndarray,
        goal: np.ndarray,
    ) -> int:
        """장애물을 배치하고 NavMesh 우회 경로 존재 여부를 확인한다.
        우회 경로가 없으면 장애물을 제거하고 -1을 반환한다."""
        # NavMesh 재계산 없이 물리 오브젝트만 배치
        prev_recompute = self._recompute
        self._recompute = False
        obj_id = self.add_obstacle(spec)
        # NavMesh 재계산 (장애물 반영)
        self._recompute_navmesh()
        self._recompute = prev_recompute

        if self._detour_exists(start, goal):
            return obj_id

        # 우회 경로 없음 → 장애물 제거
        self._sim.remove_object(obj_id)
        self._injected.remove(obj_id)
        self._recompute_navmesh()
        return -1

    def connectivity_check(
        self,
        start: np.ndarray,
        goal: np.ndarray,
    ) -> Tuple[bool, float]:
        """현재 NavMesh 기준으로 start → goal 연결성을 확인한다.

        Returns:
            (connected: bool, geodesic_dist: float)
            connected=False 이면 NavMesh가 단절된 상태 (dist=inf).
        """
        path = ShortestPath()
        path.requested_start = mn.Vector3(float(start[0]), float(start[1]), float(start[2]))
        path.requested_end   = mn.Vector3(float(goal[0]),  float(goal[1]),  float(goal[2]))
        found = self._sim.pathfinder.find_path(path)
        dist  = float(path.geodesic_distance) if found else float("inf")
        return found, dist

    def _detour_exists(self, start: np.ndarray, goal: np.ndarray) -> bool:
        """현재 NavMesh 기준으로 start → goal 경로 존재 여부를 반환한다."""
        connected, _ = self.connectivity_check(start, goal)
        return connected

    def _ensure_template(self, size: Tuple[float, float, float]) -> None:
        """크기가 달라지면 오브젝트 템플릿을 재등록한다.

        habitat_sim에 기본 등록된 'cubeSolid' 핸들을 재사용하므로
        AssetAttributesManager에 별도 등록은 불필요하다.
        """
        obj_mgr = self._sim.get_object_template_manager()
        obj_attr = obj_mgr.create_new_template(self._OBJ_TEMPLATE_NAME)
        obj_attr.render_asset_handle = self._PRIM_ASSET_HANDLE
        obj_attr.collision_asset_handle = self._PRIM_ASSET_HANDLE
        obj_attr.scale = mn.Vector3(size[0], size[1], size[2])
        obj_attr.mass = 0.0
        obj_mgr.register_template(obj_attr, self._OBJ_TEMPLATE_NAME)

    def _recompute_navmesh(self) -> bool:
        return self._sim.recompute_navmesh(
            self._sim.pathfinder,
            self._navmesh_settings,
            include_static_objects=True,
        )

    @staticmethod
    def _default_navmesh_settings() -> NavMeshSettings:
        ns = NavMeshSettings()
        ns.set_defaults()
        return ns


# ─── 벡터 환경 헬퍼 ───────────────────────────────────────────────────────────

def get_sim_from_habitat_env(env) -> object:
    """
    habitat.RLEnv / VLNCEInferenceEnv 등에서 내부 simulator를 꺼낸다.

    단일 환경(non-vectorized)만 지원한다.
    vectorized env의 경우 `envs._envs[0]`으로 먼저 꺼내야 한다.
    """
    # habitat.RLEnv → self._env.sim
    if hasattr(env, "_env") and hasattr(env._env, "sim"):
        return env._env.sim
    # habitat.Env → self.sim
    if hasattr(env, "sim"):
        return env.sim
    raise AttributeError(f"Cannot locate simulator from {type(env)}")
