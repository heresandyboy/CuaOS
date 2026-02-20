# actions.py
from __future__ import annotations

import time
from typing import Any, Dict

from src.sandbox import Sandbox
from src.config import cfg
from src.log import get_logger

log = get_logger("action")


def _pause_after_action() -> None:
    time.sleep(getattr(cfg, "PAUSE_AFTER_ACTION_SEC", getattr(cfg, "PAUSE_AFTER_CLICK_SEC", 0.25)))


def execute_action(sandbox: Sandbox, act: Dict[str, Any]) -> None:
    """Execute one action dict produced by the model."""
    a = (act.get("action") or "NOOP").upper()
    log.debug("execute: %s %s", a, {k: v for k, v in act.items() if k != "action"})

    if a == "NOOP":
        return

    if a == "WAIT":
        secs = float(act.get("seconds") or 0.5)
        time.sleep(max(0.0, min(30.0, secs)))
        return

    if a == "CLICK":
        sandbox.left_click_norm(float(act["x"]), float(act["y"]))
        _pause_after_action()
        return

    if a == "DOUBLE_CLICK":
        sandbox.double_click_norm(float(act["x"]), float(act["y"]))
        _pause_after_action()
        return

    if a == "RIGHT_CLICK":
        sandbox.right_click_norm(float(act["x"]), float(act["y"]))
        _pause_after_action()
        return

    if a == "TYPE":
        # Fara may include click_x/click_y to click a field before typing
        if "click_x" in act and "click_y" in act:
            sandbox.left_click_norm(float(act["click_x"]), float(act["click_y"]))
            _pause_after_action()
        text_val = str(act.get("text") or "")
        if act.get("delete_existing"):
            sandbox.hotkey(["ctrl", "a"])
            time.sleep(0.1)
        sandbox.type_text(text_val)
        if act.get("press_enter", False) is True:
            time.sleep(0.1)
            sandbox.press_key("enter")
        _pause_after_action()
        return

    if a == "PRESS":
        sandbox.press_key(str(act.get("key") or ""))
        _pause_after_action()
        return

    if a == "HOTKEY":
        keys = act.get("keys") or []
        sandbox.hotkey([str(k) for k in keys])
        _pause_after_action()
        return

    if a == "SCROLL":
        amount = act.get("scroll")
        if amount is None:
            amount = act.get("amount", 0)
        sandbox.scroll(int(amount or 0))
        _pause_after_action()
        return

    # Optional actions (manual / advanced)
    if a == "MOVE":
        sandbox.mouse_move_norm(float(act.get("x", 0.5)), float(act.get("y", 0.5)))
        return

    if a == "MOUSE_DOWN":
        sandbox.mouse_down(int(act.get("button", 1)))
        return

    if a == "MOUSE_UP":
        sandbox.mouse_up(int(act.get("button", 1)))
        return

    if a == "DRAG_TO":
        sandbox.drag_to_norm(
            float(act.get("x", 0.5)),
            float(act.get("y", 0.5)),
            int(act.get("button", 1)),
        )
        return

    # Compound actions (Fara-7B native actions)
    if a == "VISIT_URL":
        url = str(act.get("url", ""))
        log.info("VISIT_URL: %s", url)
        sandbox.hotkey(["ctrl", "l"])      # focus address bar
        time.sleep(0.3)
        sandbox.hotkey(["ctrl", "a"])      # select all existing text
        time.sleep(0.1)
        sandbox.type_text(url)
        time.sleep(0.1)
        sandbox.press_key("enter")
        _pause_after_action()
        return

    if a == "WEB_SEARCH":
        query = str(act.get("query", ""))
        log.info("WEB_SEARCH: %s", query)
        sandbox.hotkey(["ctrl", "l"])      # focus address bar
        time.sleep(0.3)
        sandbox.hotkey(["ctrl", "a"])      # select all existing text
        time.sleep(0.1)
        sandbox.type_text(query)
        time.sleep(0.1)
        sandbox.press_key("enter")
        _pause_after_action()
        return

    if a == "BITTI":
        return

    raise ValueError(f"Unknown action: {a} (act={act})")
