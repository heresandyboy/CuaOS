# main.py
from __future__ import annotations

import time
from typing import Any, Dict, List

from src.config import cfg, MODEL_PROFILES
from src.log import get_logger
from src.sandbox import Sandbox
from src.llm_client import load_llm, ask_next_action
from src.vision import capture_screen, draw_preview, screen_changed
from src.guards import validate_xy, check_repeat, NUDGE, STOP
from src.actions import execute_action

log = get_logger("main")


# Trim conversation history to avoid exceeding context length
def trim_history(history, keep_last=6):
    if len(history) <= keep_last:
        return history
    return history[-keep_last:]


def main() -> None:
    # Apply per-model runtime params
    _prof = MODEL_PROFILES.get(cfg.MODEL_NAME, {})
    cfg.N_CTX = _prof.get("n_ctx", cfg.N_CTX)
    cfg.N_BATCH = _prof.get("n_batch", cfg.N_BATCH)

    sandbox = Sandbox(cfg)
    sandbox.start()

    # Launch VNC viewer window (optional/noVNC)
    if getattr(cfg, "OPEN_VNC_VIEWER", True):
        sandbox.launch_vnc_viewer()

    llm = load_llm()
    log.info("cfg.N_CTX = %d", cfg.N_CTX)
    log.info("Agent ready. Type 'exit' or 'quit' to quit.")

    # Main loop: runs until user exits
    while True:
        objective = input("\nEnter command (or 'quit'): ").strip()

        # Recognize user exit
        if not objective:
            log.info("Command cannot be empty. Try again.")
            continue
        low = objective.lower()
        if low in ("exit", "quit", "q"):
            log.info("Agent shutting down.")
            break

        history: List[Dict[str, Any]] = []
        nudge_count = 0
        prev_img = None

        step = 1
        while True:
            log.info("==================== STEP %d ====================", step)

            time.sleep(cfg.WAIT_BEFORE_SCREENSHOT_SEC)

            # Capture current screenshot
            img = capture_screen(sandbox, cfg.SCREENSHOT_PATH)

            # Annotate previous action with screen-change info
            if prev_img is not None and history:
                if history[-1].get("action") != "SYSTEM_FEEDBACK":
                    changed = screen_changed(prev_img, img)
                    history[-1]["screen_changed"] = changed
                    if not changed:
                        log.info("No visible screen change after last action")

            out: Dict[str, Any] | None = None

            # Ask the model for the next action
            for attempt in range(cfg.MODEL_RETRY + 1):
                try:
                    out = ask_next_action(llm, objective, cfg.SCREENSHOT_PATH, trim_history(history))
                except Exception:
                    log.exception("ask_next_action failed (attempt %d)", attempt + 1)
                    out = None
                    continue
                action = (out.get("action") or "NOOP").upper()

                # Done statement
                if action == "BITTI":
                    log.info("BITTI -> task completed, ending loop.")
                    break

                # Normal coordinate-based actions like CLICK
                if action in ("CLICK", "DOUBLE_CLICK", "RIGHT_CLICK"):
                    x = float(out.get("x", 0.5))
                    y = float(out.get("y", 0.5))
                    ok, reason = validate_xy(x, y)
                    if ok:
                        break
                    log.warning("Invalid coordinates (%s), retrying.", reason)
                    history.append({"action": "INVALID_COORDS", "raw": out})
                    out = None
                    continue

                # Other action types are accepted directly
                break

            # If out is None, the model could not produce a valid action
            if out is None:
                log.error("Model could not produce a valid action, ending loop.")
                break

            log.info("Model output: %s", out)

            # Guard: nudge or stop
            verdict, guard_msg = check_repeat(history, out, nudge_count)
            if verdict == STOP:
                log.warning("STOP: %s", guard_msg)
                break
            if verdict == NUDGE:
                nudge_count += 1
                log.warning("NUDGE %d/%d: %s", nudge_count, cfg.MAX_NUDGES, guard_msg)
                history.append({
                    "action": "SYSTEM_FEEDBACK",
                    "target": guard_msg,
                    "why_short": f"Guard nudge #{nudge_count}",
                })
                step += 1
                if step > cfg.MAX_STEPS:
                    log.info("MAX_STEPS exceeded, ending loop.")
                    break
                continue

            # If model returned BITTI (done)
            if (out.get("action") or "").upper() == "BITTI":
                log.info("Model says task is complete. Waiting for new command.")
                break

            # Draw preview (optional)
            action = (out.get("action") or "").upper()
            if action in ("CLICK", "DOUBLE_CLICK", "RIGHT_CLICK"):
                preview_path = cfg.PREVIEW_PATH_TEMPLATE.format(i=step)
                draw_preview(img, float(out["x"]), float(out["y"]), preview_path)

            # Execute the action
            try:
                execute_action(sandbox, out)
            except Exception:
                log.exception("execute_action failed for %s", action)

            prev_img = img
            history.append(out)

            step += 1

            # Safety: stop if step count is too high
            if step > cfg.MAX_STEPS:
                log.info("MAX_STEPS exceeded, ending loop.")
                break

        # Loop for a single objective ends here,
        # then ask the user for a new command
        log.info("Ready for the next command.")


if __name__ == "__main__":
    main()
