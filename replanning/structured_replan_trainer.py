"""
StructuredVLMReplanTrainer  (v2 — temporal context)
====================================================
Training-free structured VLM replanning baseline.

Architecture:
  - NaVILA always receives the original instruction (no modification)
  - At every NaVILA mid-level decision point (queue_actions empty):
        1. NaVILA infers next mid-level action
        2. VLM receives current RGB + prev-decision RGB + context → JSON KEEP/OVERRIDE
        3. KEEP  : execute NaVILA action with full queue expansion (unchanged)
        4. OVERRIDE : execute ONE specified primitive, no queue → NaVILA decides again

VLM I/O (v2):
  Input : prev-decision RGB (if available) + current RGB (both JPEG base64)
          original instruction, NaVILA proposed action,
          last 6 NaVILA proposed actions, last 6 final executed actions,
          previous VLM decision, collision delta since last call
  Output: {"decision":"KEEP"}
          {"decision":"OVERRIDE","action":"TURN_LEFT_15"}
          (valid actions: TURN_LEFT_15 | TURN_RIGHT_15 | MOVE_FORWARD_25)

Failure policy: any API error or JSON parse error → KEEP

Log (per episode JSON):
  vlm_input_summary, navila_text, vlm_raw, vlm_decision, final_action, collision_delta
"""

import base64
import gzip
import io
import json
import os
import re
import time

import numpy as np
import requests
import torch
import tqdm
from habitat import logger
try:
    from habitat.utils.visualizations.utils import append_text_to_image
except ImportError:
    from habitat.utils.visualizations.utils import append_text_underneath_image as append_text_to_image
from habitat_baselines.common.baseline_registry import baseline_registry
try:
    from habitat_baselines.common.environments import get_env_class
except ImportError:
    from habitat.core.environments import get_env_class
from habitat_baselines.common.obs_transformers import apply_obs_transforms_batch
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
try:
    from habitat_baselines.rl.ddppo.algo.ddp_utils import is_slurm_batch_job
except ImportError:
    from habitat_baselines.rl.ddppo.ddp_utils import is_slurm_batch_job
from habitat_baselines.utils.common import batch_obs
from habitat_extensions.utils import generate_video, observations_to_image
from PIL import Image
from vlnce_baselines.common.env_utils import construct_envs_auto_reset_false
from vlnce_baselines.common.utils import extract_instruction_tokens
from vlnce_baselines.navila_trainer import NaVILATrainer, sample_and_pad_images

from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import KeywordsStoppingCriteria, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model

# ── Constants ──────────────────────────────────────────────────────────────────

_OPENROUTER_URL   = "https://openrouter.ai/api/v1/chat/completions"
_DEFAULT_VLM      = "google/gemini-2.5-flash"
_DEFAULT_TIMEOUT  = 15

_NAVILA_ACTION_NAMES = {0: "STOP", 1: "MOVE_FORWARD", 2: "TURN_LEFT", 3: "TURN_RIGHT"}

# Override action → (habitat_action_id, n_primitives)
_OVERRIDE_MAP = {
    "TURN_LEFT_15":    (2, 1),
    "TURN_RIGHT_15":   (3, 1),
    "MOVE_FORWARD_25": (1, 1),
}

_NAVILA_HIST_LEN = 6   # recent NaVILA proposed actions to track
_FINAL_HIST_LEN  = 6   # recent final executed actions to track

# ── VLM Prompts (v2) ───────────────────────────────────────────────────────────

_SR_SYSTEM = """\
You are a navigation supervisor for a mobile robot.
Decide whether to approve or override the robot's next proposed action.

Camera images provided:
  - When TWO images are given: Image 1 = previous supervisor decision view,
    Image 2 = current view. Compare them to detect movement (or lack thereof).
  - When ONE image is given: current view only (first decision of episode).

Override criteria — override ONLY when the evidence is clear:
  1. Repeated forward collisions: the robot repeatedly proposes MOVE_FORWARD
     and collisions keep occurring (collision_delta > 0, recurring over decisions).
     Override with TURN_LEFT_15 or TURN_RIGHT_15 to attempt obstacle avoidance.
  2. Rotation loop: TURN_LEFT and TURN_RIGHT alternate in the action history
     without any MOVE_FORWARD, AND the two camera views look similar
     (robot is spinning in place without progress).
     Override with MOVE_FORWARD_25 to break the loop — only if the view ahead
     looks passable.
  3. Default to KEEP when evidence is insufficient or ambiguous.

Do NOT use goal positions, maps, coordinates, or any privileged information.
Do NOT override based on visual uncertainty alone.

Reply with ONLY valid JSON on a single line, no markdown, no explanation:
{"decision":"KEEP"}
or
{"decision":"OVERRIDE","action":"TURN_LEFT_15"}

Valid override actions: TURN_LEFT_15, TURN_RIGHT_15, MOVE_FORWARD_25\
"""

_SR_USER_TMPL = """\
Navigation instruction: "{orig_instr}"

Robot's proposed action: "{navila_text}" → {action_name}
Recent NaVILA proposed actions (oldest→newest, last {hist_len}): {navila_hist}
Recent final executed actions (oldest→newest, last {hist_len}): {final_hist}
Previous supervisor decision: {prev_vlm}
Collisions recorded since last decision: {collision_delta}

What is your decision?\
"""


# ── Trainer ────────────────────────────────────────────────────────────────────

@baseline_registry.register_trainer(name="navila_structured_replan")
class StructuredVLMReplanTrainer(NaVILATrainer):
    """
    NaVILA에는 항상 original instruction만 전달.
    NaVILA mid-level 예측 시점마다 VLM이 KEEP/OVERRIDE 판단.
    OVERRIDE는 지정 primitive 1회 실행 후 즉시 NaVILA 복귀.
    """

    # ── VLM helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _encode_rgb(rgb_np: np.ndarray) -> str:
        img = Image.fromarray(rgb_np.astype(np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode()

    def _call_vlm(
        self,
        curr_rgb_np: np.ndarray,
        prev_rgb_np,              # np.ndarray or None
        orig_instr: str,
        navila_text: str,
        navila_action_id: int,
        navila_hist: list,
        final_hist: list,
        prev_vlm_dec: dict,
        collision_delta: int,
        vlm_model: str,
        timeout: int,
    ) -> dict:
        """
        Returns {"decision":"KEEP"} or {"decision":"OVERRIDE","action":"..."}
        plus "_raw" and optional "_error" metadata fields.
        Any failure returns KEEP.
        Two images sent when prev_rgb_np is not None (prev + curr);
        one image sent on first decision of each episode (curr only).
        """
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        raw_output = ""

        if not api_key:
            logger.warning("[SRTrainer] OPENROUTER_API_KEY not set → KEEP")
            return {"decision": "KEEP", "_raw": "", "_error": "no_api_key"}

        try:
            b64_curr = self._encode_rgb(curr_rgb_np)
            action_name = _NAVILA_ACTION_NAMES.get(navila_action_id, "UNKNOWN")

            prev_str = prev_vlm_dec.get("decision", "KEEP")
            if prev_vlm_dec.get("action"):
                prev_str += f"({prev_vlm_dec['action']})"

            user_text = _SR_USER_TMPL.format(
                orig_instr=orig_instr,
                navila_text=navila_text,
                action_name=action_name,
                hist_len=_NAVILA_HIST_LEN,
                navila_hist=navila_hist if navila_hist else [],
                final_hist=final_hist if final_hist else [],
                prev_vlm=prev_str,
                collision_delta=collision_delta,
            )

            # Build image content: prev (if available) then curr
            content = [{"type": "text", "text": user_text}]
            if prev_rgb_np is not None:
                b64_prev = self._encode_rgb(prev_rgb_np)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_prev}", "detail": "low"},
                })
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_curr}", "detail": "low"},
            })

            payload = {
                "model": vlm_model,
                "max_tokens": 50,
                "temperature": 0.0,
                "messages": [
                    {"role": "system", "content": _SR_SYSTEM},
                    {"role": "user",   "content": content},
                ],
            }

            resp = requests.post(
                _OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=timeout,
            )
            resp.raise_for_status()
            raw_output = resp.json()["choices"][0]["message"]["content"].strip()

            # Strip markdown fences if present
            clean = raw_output
            if clean.startswith("```"):
                clean = re.sub(r"^```[a-z]*\n?", "", clean)
                clean = re.sub(r"\n?```$", "", clean)
            parsed = json.loads(clean.strip())

            decision = parsed.get("decision", "").upper()
            if decision not in ("KEEP", "OVERRIDE"):
                raise ValueError(f"unexpected decision value: {decision!r}")

            if decision == "OVERRIDE":
                ov_action = parsed.get("action", "").upper()
                if ov_action not in _OVERRIDE_MAP:
                    raise ValueError(f"unexpected override action: {ov_action!r}")
                logger.info(f"[SRTrainer] VLM → OVERRIDE({ov_action})")
                return {"decision": "OVERRIDE", "action": ov_action, "_raw": raw_output}

            logger.info("[SRTrainer] VLM → KEEP")
            return {"decision": "KEEP", "_raw": raw_output}

        except Exception as exc:
            logger.warning(f"[SRTrainer] VLM call/parse failed: {exc} → KEEP")
            return {"decision": "KEEP", "_raw": raw_output, "_error": str(exc)}

    # ── Main eval loop ─────────────────────────────────────────────────────────

    def _eval_checkpoint(self, checkpoint_path: str, writer: TensorboardWriter) -> None:
        logger.info(f"checkpoint_path: {checkpoint_path}")

        model_name = os.path.basename(os.path.normpath(checkpoint_path))
        tokenizer, model, image_processor, _ = load_pretrained_model(checkpoint_path, model_name)
        model = model.eval()

        config = self.config.clone()
        split = config.EVAL.SPLIT

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = split
        config.TASK_CONFIG.DATASET.ROLES = ["guide"]
        config.TASK_CONFIG.DATASET.LANGUAGES = config.EVAL.LANGUAGES
        config.TASK_CONFIG.TASK.NDTW.SPLIT = split
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        config.TASK_CONFIG.DATASET.NUM_CHUNKS = self.num_chunks
        config.TASK_CONFIG.DATASET.CHUNK_IDX = self.chunk_idx
        config.RESULTS_DIR = os.path.join(
            config.RESULTS_DIR, model_name,
            config.TASK_CONFIG.DATASET.TYPE,
            config.TASK_CONFIG.DATASET.SPLIT,
        )
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        config.VIDEO_DIR = os.path.join(config.RESULTS_DIR, "videos")
        config.use_pbar = not is_slurm_batch_job()
        if len(config.VIDEO_OPTION) > 0:
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
        config.freeze()

        # Skip if already done
        if config.EVAL.SAVE_RESULTS:
            fname = os.path.join(
                config.RESULTS_DIR,
                f"{split}_{self.num_chunks}-{self.chunk_idx}.json",
            )
            if os.path.exists(fname):
                logger.info("skipping -- evaluation exists.")
                return

        # Structured Replan config
        sr_cfg = getattr(self.config, "STRUCTURED_REPLAN", None)
        vlm_model   = getattr(sr_cfg, "VLM_MODEL", _DEFAULT_VLM)     if sr_cfg else _DEFAULT_VLM
        vlm_timeout = int(getattr(sr_cfg, "TIMEOUT", _DEFAULT_TIMEOUT)) if sr_cfg else _DEFAULT_TIMEOUT
        log_dir = os.path.join(config.RESULTS_DIR, "vlm_logs")
        os.makedirs(log_dir, exist_ok=True)
        logger.info(f"[SRTrainer] VLM={vlm_model} timeout={vlm_timeout}s log_dir={log_dir}")

        # Env + data
        envs = construct_envs_auto_reset_false(config, get_env_class(config.ENV_NAME))
        observations = envs.reset()
        observations = extract_instruction_tokens(
            observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
        )
        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        stats_episodes = {}
        past_rgbs  = [[] for _ in range(envs.num_envs)]
        rgb_frames = [[] for _ in range(envs.num_envs)]

        if len(config.VIDEO_OPTION) > 0:
            os.makedirs(config.VIDEO_DIR, exist_ok=True)

        num_eps = sum(envs.number_of_episodes)
        if config.EVAL.EPISODE_COUNT > -1:
            num_eps = min(config.EVAL.EPISODE_COUNT, num_eps)

        pbar = tqdm.tqdm(total=num_eps) if config.use_pbar else None
        start_time = time.time()
        assert envs.num_envs == 1

        # NaVILA action parser
        _patterns = {
            0: re.compile(r"\bstop\b",            re.IGNORECASE),
            1: re.compile(r"\bis move forward\b", re.IGNORECASE),
            2: re.compile(r"\bis turn left\b",    re.IGNORECASE),
            3: re.compile(r"\bis turn right\b",   re.IGNORECASE),
        }

        def _parse_navila(text: str) -> int:
            for act, pat in _patterns.items():
                if pat.search(text):
                    return act
            return 1  # default forward

        # ── Per-episode state ──────────────────────────────────────────────────
        queue_actions  = []
        _ep_ready      = False   # True once episode state is initialized
        _orig_instr    = ""
        _prev_vlm      = {"decision": "KEEP"}
        _prev_vlm_rgb  = None    # RGB array at previous VLM decision point
        _navila_hist   = []      # last _NAVILA_HIST_LEN NaVILA proposed action texts
        _final_hist    = []      # last _FINAL_HIST_LEN final executed action names
        _coll_delta    = 0       # collision count accumulated since last VLM call
        _last_coll     = 0       # cumulative collision count at last step
        _hab_step      = 0       # habitat primitive step counter
        _ep_log        = []      # decision log for this episode
        _vlm_calls     = 0
        _keep_count    = 0
        _ov_count      = 0
        _ov_breakdown  = {k: 0 for k in _OVERRIDE_MAP}

        def _init_episode():
            nonlocal _ep_ready, _orig_instr, _prev_vlm, _prev_vlm_rgb
            nonlocal _navila_hist, _final_hist
            nonlocal _coll_delta, _last_coll, _hab_step, _ep_log
            nonlocal _vlm_calls, _keep_count, _ov_count, _ov_breakdown
            _orig_instr   = envs.current_episodes()[0].instruction.instruction_text
            _prev_vlm     = {"decision": "KEEP"}
            _prev_vlm_rgb = None
            _navila_hist  = []
            _final_hist   = []
            _coll_delta   = 0
            _last_coll    = 0
            _hab_step     = 0
            _ep_log       = []
            _vlm_calls    = 0
            _keep_count   = 0
            _ov_count     = 0
            _ov_breakdown = {k: 0 for k in _OVERRIDE_MAP}
            _ep_ready     = True
            ep_id = envs.current_episodes()[0].episode_id
            logger.info(f"[SRTrainer] ep={ep_id} init | instr='{_orig_instr[:80]}'")

        def _save_episode_log(ep_id, final_info: dict):
            out = {
                "episode_id":       str(ep_id),
                "success":          final_info.get("success", 0),
                "spl":              final_info.get("spl", 0.0),
                "ndtw":             final_info.get("ndtw", 0.0),
                "steps_taken":      final_info.get("steps_taken", _hab_step),
                "collision_count":  final_info.get("collision_count", 0),
                "vlm_calls":        _vlm_calls,
                "keep_count":       _keep_count,
                "override_count":   _ov_count,
                "override_breakdown": dict(_ov_breakdown),
                "navila_hist_len":  _NAVILA_HIST_LEN,
                "decisions":        _ep_log,
            }
            path = os.path.join(log_dir, f"ep_{ep_id}.json")
            with open(path, "w") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)
            logger.info(f"[SRTrainer] log saved → {path}")
            return path

        # Init first episode
        _init_episode()

        # ── Main loop ──────────────────────────────────────────────────────────
        while envs.num_envs > 0 and len(stats_episodes) < num_eps:
            current_episodes = envs.current_episodes()

            if len(queue_actions) > 0:
                # ── [A] Queue drain: execute queued primitive, no VLM ─────────
                print(f"using queue...{queue_actions[0]}")
                outputs = envs.step([queue_actions[0]])
                queue_actions.pop(0)
                print(f"queue length after using...{len(queue_actions)}")

            else:
                # ── [B] NaVILA mid-level decision point ★ ────────────────────
                with torch.no_grad():
                    curr_rgb_np  = batch[0]["rgb"].cpu().numpy()
                    curr_rgb_img = Image.fromarray(np.uint8(curr_rgb_np)).convert("RGB")

                    frames = sample_and_pad_images(
                        past_rgbs[0] + [curr_rgb_img],
                        num_frames=model.config.num_video_frames,
                    )
                    print(f"input frame length {len(frames)}")

                    # NaVILA always gets original instruction
                    instruction   = current_episodes[0].instruction.instruction_text
                    interleaved   = "<image>\n" * (len(frames) - 1)
                    question = (
                        "Imagine you are a robot programmed for navigation tasks. You have been given a video "
                        f"of historical observations {interleaved}, and current observation <image>\n. "
                        f'Your assigned task is: "{instruction}" '
                        "Analyze this series of images to decide your next action, which could be turning left "
                        "or right by a specific degree, moving forward a certain distance, or stop if the task "
                        "is completed."
                    )

                    conv = conv_templates["llama_3"].copy()
                    conv.append_message(conv.roles[0], question)
                    conv.append_message(conv.roles[1], None)
                    prompt = conv.get_prompt()

                    images_tensor = process_images(frames, image_processor, model.config).to(
                        model.device, dtype=torch.float16
                    )
                    input_ids = (
                        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                        .unsqueeze(0).cuda()
                    )
                    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

                    with torch.inference_mode():
                        output_ids = model.generate(
                            input_ids,
                            images=images_tensor.half().cuda(),
                            do_sample=False,
                            temperature=0.0,
                            max_new_tokens=32,
                            use_cache=True,
                            stopping_criteria=[stopping_criteria],
                            pad_token_id=tokenizer.eos_token_id,
                        )

                navila_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                if navila_text.endswith(stop_str):
                    navila_text = navila_text[:-len(stop_str)].strip()
                print(navila_text)

                navila_action_id = _parse_navila(navila_text)
                print([navila_action_id])

                # ── VLM intervention ──────────────────────────────────────────
                coll_for_vlm  = _coll_delta
                prev_rgb_snap = _prev_vlm_rgb   # RGB from last VLM call (None on first)
                vlm_resp = self._call_vlm(
                    curr_rgb_np      = curr_rgb_np,
                    prev_rgb_np      = prev_rgb_snap,
                    orig_instr       = _orig_instr,
                    navila_text      = navila_text,
                    navila_action_id = navila_action_id,
                    navila_hist      = list(_navila_hist),
                    final_hist       = list(_final_hist),
                    prev_vlm_dec     = {k: v for k, v in _prev_vlm.items() if not k.startswith("_")},
                    collision_delta  = coll_for_vlm,
                    vlm_model        = vlm_model,
                    timeout          = vlm_timeout,
                )
                # Save current RGB as "previous" for next VLM call
                _prev_vlm_rgb = curr_rgb_np.copy()
                _coll_delta   = 0   # reset accumulator
                _vlm_calls   += 1
                _prev_vlm     = {k: v for k, v in vlm_resp.items() if not k.startswith("_")}

                # Update NaVILA proposed action history (keep last _NAVILA_HIST_LEN)
                _navila_hist.append(navila_text)
                if len(_navila_hist) > _NAVILA_HIST_LEN:
                    _navila_hist.pop(0)

                decision = vlm_resp["decision"]
                print(
                    f"[SRTrainer] hab_step={_hab_step} VLM={decision} "
                    f"navila={_NAVILA_ACTION_NAMES.get(navila_action_id,'?')} | "
                    f"{navila_text[:70]}"
                )

                # ── Execute ───────────────────────────────────────────────────
                if decision == "OVERRIDE":
                    ov_name   = vlm_resp.get("action", "TURN_LEFT_15")
                    ov_act_id, _ = _OVERRIDE_MAP.get(ov_name, (2, 1))
                    final_action_id   = ov_act_id
                    final_action_name = ov_name
                    _ov_count  += 1
                    _ov_breakdown[ov_name] = _ov_breakdown.get(ov_name, 0) + 1
                    print(f"[SRTrainer] OVERRIDE → {ov_name} (id={ov_act_id})")
                    outputs = envs.step([ov_act_id])
                    # queue stays empty → NaVILA decides again next iter

                else:
                    # KEEP: NaVILA action with full queue expansion
                    _keep_count       += 1
                    final_action_id    = navila_action_id
                    final_action_name  = _NAVILA_ACTION_NAMES.get(navila_action_id, "?")

                    if navila_action_id == 1:
                        try:
                            m = re.search(r"move forward (\d+) cm", navila_text)
                            distance = int(m.group(1))
                        except Exception:
                            distance = 25
                        if (distance % 25) != 0:
                            distance = min([25, 50, 75], key=lambda x: abs(x - distance))
                        outputs = envs.step([1])
                        for _ in range(int(distance // 25) - 1):
                            queue_actions.append(1)

                    elif navila_action_id == 2:
                        try:
                            m = re.search(r"turn left (\d+) degree", navila_text)
                            degree = int(m.group(1))
                        except Exception:
                            degree = 15
                        if (degree % 15) != 0:
                            degree = min([15, 30, 45], key=lambda x: abs(x - degree))
                        outputs = envs.step([2])
                        for _ in range(int(degree // 15) - 1):
                            queue_actions.append(2)
                        print(f"queue length: {len(queue_actions)}")

                    elif navila_action_id == 3:
                        try:
                            m = re.search(r"turn right (\d+) degree", navila_text)
                            degree = int(m.group(1))
                        except Exception:
                            degree = 15
                        if (degree % 15) != 0:
                            degree = min([15, 30, 45], key=lambda x: abs(x - degree))
                        outputs = envs.step([3])
                        for _ in range(int(degree // 15) - 1):
                            queue_actions.append(3)

                    else:  # STOP
                        outputs = envs.step([navila_action_id])

                # Update final executed action history (keep last _FINAL_HIST_LEN)
                _final_hist.append(final_action_name)
                if len(_final_hist) > _FINAL_HIST_LEN:
                    _final_hist.pop(0)

                # Log this decision
                _ep_log.append({
                    "hab_step":     _hab_step,
                    "has_prev_rgb": prev_rgb_snap is not None,
                    "vlm_input_summary": {
                        "orig_instr":      _orig_instr[:80],
                        "navila_text":     navila_text,
                        "navila_action":   _NAVILA_ACTION_NAMES.get(navila_action_id, "?"),
                        "navila_hist":     list(_navila_hist[:-1]),   # before this text was appended
                        "final_hist":      list(_final_hist[:-1]),    # before this action was appended
                        "prev_vlm":        {k: v for k, v in _prev_vlm.items() if not k.startswith("_")},
                        "collision_delta": coll_for_vlm,
                    },
                    "vlm_raw":           vlm_resp.get("_raw", ""),
                    "vlm_decision":      decision,
                    "vlm_override_action": vlm_resp.get("action", ""),
                    "navila_action_id":  navila_action_id,
                    "final_action_id":   final_action_id,
                    "final_action_name": final_action_name,
                    "collision_delta":   coll_for_vlm,
                })

            # ── Unpack outputs ────────────────────────────────────────────────
            observations, _, dones, infos = [list(x) for x in zip(*outputs)]
            _hab_step += 1

            # Accumulate collision delta
            curr_coll    = int(infos[0].get("collision_count", 0))
            _coll_delta += curr_coll - _last_coll
            _last_coll   = curr_coll

            for i in range(envs.num_envs):
                past_rgbs[i].append(
                    Image.fromarray(batch[0]["rgb"].cpu().numpy()).convert("RGB")
                )
                if infos[i].get("flush_queue", False):
                    past_rgbs[i] = []
                    queue_actions = []
                    print("[Trainer] flush_queue: cleared (SmartShield REJOIN DONE)")

                if len(config.VIDEO_OPTION) > 0:
                    frame = observations_to_image(observations[i], infos[i])
                    frame = append_text_to_image(
                        frame, current_episodes[i].instruction.instruction_text
                    )
                    rgb_frames[i].append(frame)

                if not dones[i]:
                    continue

                # ── Episode done ──────────────────────────────────────────────
                ep_id = current_episodes[i].episode_id
                stats_episodes[ep_id] = infos[i]

                print(
                    f"[SRTrainer] ep={ep_id} DONE | "
                    f"success={infos[i].get('success',0):.0f} "
                    f"spl={infos[i].get('spl',0):.3f} "
                    f"ndtw={infos[i].get('ndtw',0):.3f} "
                    f"steps={infos[i].get('steps_taken',_hab_step)} "
                    f"coll={infos[i].get('collision_count',0)} | "
                    f"vlm_calls={_vlm_calls} keep={_keep_count} override={_ov_count} "
                    f"ov_breakdown={_ov_breakdown}"
                )

                _save_episode_log(ep_id, infos[i])

                observations[i] = envs.reset_at(i)[0]
                past_rgbs[i]    = []
                queue_actions   = []
                _ep_ready       = False
                _prev_vlm_rgb   = None   # clear between episodes

                if pbar:
                    pbar.update()
                else:
                    logger.info(
                        f"[Ckpt: {checkpoint_path}] "
                        f"[Episodes: {len(stats_episodes)}/{num_eps}] "
                        f"[Time: {round(time.time()-start_time)}s]"
                    )

                if len(config.VIDEO_OPTION) > 0:
                    generate_video(
                        video_option=config.VIDEO_OPTION,
                        video_dir=config.VIDEO_DIR,
                        images=rgb_frames[i],
                        episode_id=ep_id,
                        checkpoint_idx="0",
                        metrics={"spl": stats_episodes[ep_id].get("spl", float("nan"))},
                        tb_writer=writer,
                    )
                    stats_episodes[ep_id].pop("top_down_map_vlnce", None)
                    rgb_frames[i] = []

            observations = extract_instruction_tokens(
                observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
            )
            batch = batch_obs(observations, self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

            envs_to_pause = []
            next_episodes = envs.current_episodes()
            for i in range(envs.num_envs):
                if next_episodes[i].episode_id in stats_episodes:
                    envs_to_pause.append(i)
                elif not _ep_ready:
                    _init_episode()

            envs, batch, rgb_frames = self._pause_envs(
                envs_to_pause, envs, batch, rgb_frames,
            )

        envs.close()
        if pbar:
            pbar.close()

        if config.EVAL.SAVE_RESULTS:
            with open(fname, "w") as f:
                json.dump(stats_episodes, f, indent=4)
