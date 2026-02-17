# agent_runner_v2.py — Plan-based agent execution
from __future__ import annotations

import time
import threading
from typing import Any, Dict, List, Optional, Callable, Tuple

from src.config import cfg
from src.sandbox import Sandbox
from src.vision import capture_screen, draw_preview
from src.guards import validate_xy, should_stop_on_repeat
from src.actions import execute_action
from src.llm_client import ask_next_action


# ─── History Helpers ──────────────────────────────────────────────────

def trim_history(history: List[Dict[str, Any]], keep_last: int = 6) -> List[Dict[str, Any]]:
    return history[-keep_last:] if len(history) > keep_last else history


def _center_from_bbox(b: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = map(float, b)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _extract_xy(out: Dict[str, Any]) -> Tuple[float, float]:
    x = out.get("x", 0.5)
    y = out.get("y", 0.5)
    pos = out.get("position", None)
    if pos is not None and isinstance(pos, (list, tuple)):
        if len(pos) == 2 and all(isinstance(t, (int, float)) for t in pos):
            return float(pos[0]), float(pos[1])
        if len(pos) == 4 and all(isinstance(t, (int, float)) for t in pos):
            return _center_from_bbox(list(pos))
        if len(pos) == 2 and all(isinstance(t, (list, tuple)) and len(t) == 2 for t in pos):
            (x1, y1), (x2, y2) = pos
            return (float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0
    if isinstance(x, (list, tuple)):
        if len(x) == 2: return float(x[0]), float(x[1])
        if len(x) == 4: return _center_from_bbox(list(x))
    if isinstance(y, (list, tuple)):
        if len(y) == 2: return float(y[0]), float(y[1])
        if len(y) == 4: return _center_from_bbox(list(y))
    return float(x), float(y)


# ─── Single Sub-Step Runner ──────────────────────────────────────────

def run_single_substep(
    sandbox: Sandbox,
    llm,
    objective: str,
    log: Optional[Callable[[str, str], None]] = None,
    stop_event: Optional[threading.Event] = None,
    max_steps: int = 10,
) -> str:
    """
    Execute a single sub-step objective (e.g. "click browser icon on taskbar")
    using Qwen3-VL. Returns a result string.
    """
    def _log(msg: str, level: str = "info"):
        if log:
            log(msg, level)

    history: List[Dict[str, Any]] = []
    step = 1

    while True:
        if stop_event and stop_event.is_set():
            return "STOPPED"

        _log(f"    [Step {step}]", "info")
        time.sleep(getattr(cfg, "WAIT_BEFORE_SCREENSHOT_SEC", 0.1))
        img = capture_screen(sandbox, cfg.SCREENSHOT_PATH)

        out: Optional[Dict[str, Any]] = None

        for attempt in range(getattr(cfg, "MODEL_RETRY", 2) + 1):
            out = ask_next_action(llm, objective, cfg.SCREENSHOT_PATH, trim_history(history))
            action = (out.get("action") or "NOOP").upper()

            if action == "BITTI":
                return "DONE"

            if action in ("CLICK", "DOUBLE_CLICK", "RIGHT_CLICK"):
                x, y = _extract_xy(out)
                ok, reason = validate_xy(x, y)
                if ok:
                    out["x"], out["y"] = x, y
                    break
                _log(f"    [WARN] Invalid coordinates ({reason}), retrying.", "warn")
                history.append({"action": "INVALID_COORDS", "raw": out})
                out = None
                continue

            # Other action types accepted
            break

        if out is None:
            return "ERROR(no valid action)"

        action = (out.get("action") or "").upper()
        detail = out.get("why_short", out.get("target", ""))
        _log(f"    [MODEL] {action}: {detail}", "model")

        stop, why = should_stop_on_repeat(history, out)
        if stop:
            _log(f"    [STOPPED] {why}", "warn")
            return "DONE(repeat-guard)"

        if action in ("CLICK", "DOUBLE_CLICK", "RIGHT_CLICK"):
            preview_path = cfg.PREVIEW_PATH_TEMPLATE.format(i=step)
            draw_preview(img, float(out["x"]), float(out["y"]), preview_path)

        execute_action(sandbox, out)
        history.append(out)

        step += 1
        if step > max_steps:
            return "DONE(max-substeps)"


# ─── Plan-Based Command Runner ───────────────────────────────────────

def run_planned_command(
    sandbox: Sandbox,
    llm,
    plan_steps: List[str],
    log: Optional[Callable[[str, str], None]] = None,
    stop_event: Optional[threading.Event] = None,
) -> str:
    """
    Execute a list of planned sub-steps sequentially.
    Each sub-step becomes the objective for Qwen3-VL.
    """
    def _log(msg: str, level: str = "info"):
        if log:
            log(msg, level)

    total = len(plan_steps)
    _log(f"═══ EXECUTING PLAN ({total} steps) ═══", "info")

    for i, step_text in enumerate(plan_steps, 1):
        if stop_event and stop_event.is_set():
            return "STOPPED"

        _log(f"", "info")
        _log(f"══ PLAN STEP {i}/{total}: {step_text} ══", "info")

        # Use the sub-step text as the objective for Qwen3-VL
        result = run_single_substep(
            sandbox=sandbox,
            llm=llm,
            objective=step_text,
            log=log,
            stop_event=stop_event,
            max_steps=getattr(cfg, "MAX_STEPS", 20),
        )

        _log(f"  → Result: {result}", "success" if "DONE" in result else "warn")

        if result == "STOPPED":
            return "STOPPED"
        if result.startswith("ERROR"):
            _log(f"  Sub-step failed, continuing to next step...", "warn")

    return "DONE(plan-complete)"
