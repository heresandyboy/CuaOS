# guards.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from src.config import cfg
from src.log import get_logger

log = get_logger("guard")

CLICK_ACTIONS = {"CLICK", "DOUBLE_CLICK", "RIGHT_CLICK"}

# Wider tolerance for "same region" (e.g. clicking around the same UI element)
_REGION_EPS = 0.05  # 5% of screen — catches repeated clicks on the same button/tab

# Verdict constants returned by check_repeat()
OK = "ok"
NUDGE = "nudge"
STOP = "stop"


def validate_xy(x: float, y: float) -> Tuple[bool, str]:
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        return False, "x/y out of [0,1]"
    if x < cfg.MIN_MARGIN or x > (1.0 - cfg.MIN_MARGIN) or y < cfg.MIN_MARGIN or y > (1.0 - cfg.MIN_MARGIN):
        return False, f"x/y too close to edge (margin={cfg.MIN_MARGIN})"
    return True, ""


def action_signature(act: Dict[str, Any]) -> str:
    a = (act.get("action") or "NOOP").upper()
    if a in CLICK_ACTIONS:
        return f"{a}:{float(act.get('x', 0)):.4f},{float(act.get('y', 0)):.4f}"
    if a == "TYPE":
        return f"TYPE:{act.get('text','')}"
    if a == "PRESS":
        return f"PRESS:{act.get('key','')}"
    if a == "HOTKEY":
        return f"HOTKEY:{','.join(act.get('keys') or [])}"
    if a == "SCROLL":
        return f"SCROLL:{int(act.get('scroll',0))}"
    if a == "WAIT":
        return f"WAIT:{float(act.get('seconds',0))}"
    if a == "VISIT_URL":
        return f"VISIT_URL:{act.get('url','')}"
    if a == "WEB_SEARCH":
        return f"WEB_SEARCH:{act.get('query','')}"
    return a


def _same_xy(a, b, eps: float) -> bool:
    try:
        return abs(float(a["x"]) - float(b["x"])) <= eps and abs(float(a["y"]) - float(b["y"])) <= eps
    except Exception:
        return False


def _is_click(act: Dict[str, Any]) -> bool:
    return (act.get("action") or "").upper() in CLICK_ACTIONS


def _real_actions(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter out SYSTEM_FEEDBACK entries so detection windows only count real actions."""
    return [h for h in history if (h.get("action") or "").upper() != "SYSTEM_FEEDBACK"]


def _actions_since_last_nudge(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return only real actions since the most recent SYSTEM_FEEDBACK (nudge).

    If no nudge has been given, returns all real actions.
    """
    # Walk backwards to find the last SYSTEM_FEEDBACK
    last_nudge_idx = -1
    for i in range(len(history) - 1, -1, -1):
        if (history[i].get("action") or "").upper() == "SYSTEM_FEEDBACK":
            last_nudge_idx = i
            break

    if last_nudge_idx < 0:
        return _real_actions(history)

    # Only look at actions AFTER the last nudge
    return _real_actions(history[last_nudge_idx + 1:])


def _model_changed_approach(history: List[Dict[str, Any]], new_action: Dict[str, Any]) -> bool:
    """Check if the model switched to a fundamentally different action type after a nudge.

    Returns True if the last history entry is SYSTEM_FEEDBACK and the new action
    is a different category (e.g., keyboard instead of click).
    """
    if not history:
        return False

    last = history[-1]
    if (last.get("action") or "").upper() != "SYSTEM_FEEDBACK":
        return False

    new_type = (new_action.get("action") or "").upper()

    # Check what the last real action was (before the feedback)
    real = _real_actions(history)
    if not real:
        return True  # no previous action to compare

    last_real_type = (real[-1].get("action") or "").upper()

    # "Changed approach" = different action category
    click_types = CLICK_ACTIONS
    keyboard_types = {"TYPE", "PRESS", "HOTKEY"}
    nav_types = {"SCROLL", "WAIT", "VISIT_URL", "WEB_SEARCH"}

    def category(t: str) -> str:
        if t in click_types:
            return "click"
        if t in keyboard_types:
            return "keyboard"
        if t in nav_types:
            return "nav"
        return t

    changed = category(new_type) != category(last_real_type)
    if changed:
        log.info("Model changed approach: %s -> %s (allowing through)", last_real_type, new_type)
    return changed


def _detect_oscillation(history: List[Dict[str, Any]], new_action: Dict[str, Any],
                        nudge_count: int = 0, window: int = 6) -> Tuple[bool, str]:
    """Detect A->B->A->B oscillation patterns in the recent history.

    Looks at the last `window` real actions + new_action.  If all clicks in
    that window land in only 1-2 distinct regions, the model is stuck oscillating.
    """
    eps = _REGION_EPS
    real = _real_actions(history)
    recent = list(real[-window:]) + [new_action]
    clicks = [a for a in recent if _is_click(a)]

    if len(clicks) < 4:
        return False, ""

    regions: List[Dict[str, Any]] = []
    for c in clicks:
        cx, cy = float(c.get("x", 0)), float(c.get("y", 0))
        matched = False
        for r in regions:
            if abs(cx - r["x"]) <= eps and abs(cy - r["y"]) <= eps:
                r["count"] += 1
                matched = True
                break
        if not matched:
            regions.append({"x": cx, "y": cy, "count": 1})

    if len(regions) <= 2 and len(clicks) >= 5:
        if nudge_count == 0:
            msg = (
                f"You are stuck in a loop: your last {len(clicks)} clicks have all "
                f"landed in the same {len(regions)} area(s). The UI is not responding. "
                "Try a completely different approach — use keyboard shortcuts, "
                "scroll to find other elements, or interact with a different part of the screen."
            )
        else:
            msg = (
                f"STILL stuck clicking the same {len(regions)} spot(s). Clicking is NOT working. "
                "STOP clicking and use keyboard actions instead: "
                "hotkey(key='ctrl l') to focus address bar, type(content='...') to enter text, "
                "or scroll(direction='down') to see more content."
            )
        return True, msg

    return False, ""


def _detect_no_progress(history: List[Dict[str, Any]], new_action: Dict[str, Any],
                        nudge_count: int = 0, window: int = 6) -> Tuple[bool, str]:
    """Detect when recent actions have all been marked as 'no visible change'.

    After a nudge, only looks at actions SINCE that nudge to give the model
    a fair chance to recover.
    """
    # After a nudge, only look at post-nudge actions
    if nudge_count > 0:
        real = _actions_since_last_nudge(history)
    else:
        real = _real_actions(history)

    if len(real) < window:
        return False, ""

    recent = real[-window:]
    no_change_count = sum(1 for a in recent if a.get("screen_changed") is False)

    if no_change_count >= window:
        # Escalate the message based on how many nudges have already been given
        if nudge_count == 0:
            msg = (
                f"Your last {no_change_count} actions produced NO visible change on screen. "
                "The clicks are not landing on interactive elements. "
                "Try using keyboard shortcuts (e.g., hotkey key='ctrl l' to focus the address bar, "
                "then type a URL), or scroll to reveal hidden content."
            )
        elif nudge_count == 1:
            msg = (
                f"STILL no progress after {no_change_count} failed actions. "
                "Clicking is NOT working. You MUST use a keyboard-based approach: "
                "use hotkey(key='ctrl l') to focus the browser address bar, "
                "then type(content='https://...') to navigate directly. "
                "Or use hotkey(key='ctrl t') to open a new tab."
            )
        else:
            msg = (
                f"FINAL WARNING: {no_change_count} consecutive actions with zero effect. "
                "Stop clicking. Use ONLY keyboard actions: "
                "hotkey(key='ctrl l') then type(content='your URL here'). "
                "This is your last chance before the task is terminated."
            )
        return True, msg

    return False, ""


def _detect_direct_repeat(history: List[Dict[str, Any]], new_action: Dict[str, Any]) -> Tuple[bool, str]:
    """Detect direct A->A->A repetition patterns."""
    real = _real_actions(history)
    if not real:
        return False, ""

    last = real[-1]
    a1 = (last.get("action") or "").upper()
    a2 = (new_action.get("action") or "").upper()

    if a1 != a2:
        return False, ""

    # TYPE: 3 consecutive identical texts
    if a2 == "TYPE":
        if len(real) >= 2:
            prev2 = real[-2]
            if ((prev2.get("action") or "").upper() == "TYPE"
                    and (prev2.get("text") or "") == (last.get("text") or "")
                    and (last.get("text") or "") == (new_action.get("text") or "")):
                return True, (
                    f"You are trying to type '{new_action.get('text', '')[:40]}' for the 3rd time in a row. "
                    "This text has already been entered. Check if the input field already has your text, "
                    "or try clicking on the correct input field first before typing."
                )

    # PRESS: 2 consecutive identical
    if a2 == "PRESS":
        if (last.get("key") or "") == (new_action.get("key") or ""):
            return True, (
                f"You are pressing '{new_action.get('key', '')}' again with no effect. "
                "Try a different key or a different approach entirely."
            )

    # HOTKEY: 2 consecutive identical
    if a2 == "HOTKEY":
        if (last.get("keys") or []) == (new_action.get("keys") or []):
            keys_str = "+".join(new_action.get("keys") or [])
            return True, (
                f"You are pressing '{keys_str}' again with no effect. "
                "Try a different shortcut or interact with the UI directly."
            )

    # CLICK: 3 consecutive at exact same coords
    if a2 in CLICK_ACTIONS:
        eps = float(getattr(cfg, "REPEAT_XY_EPS", 0.01))
        if len(real) >= 2:
            prev2 = real[-2]
            if (_is_click(prev2) and _same_xy(prev2, new_action, eps)
                    and _is_click(last) and _same_xy(last, new_action, eps)):
                return True, (
                    "You have clicked the exact same spot 3 times in a row. "
                    "The element is not responding as expected. Try clicking somewhere else, "
                    "double-clicking, right-clicking, or using keyboard navigation instead."
                )

    return False, ""


def check_repeat(history: List[Dict[str, Any]], new_action: Dict[str, Any],
                 nudge_count: int = 0) -> Tuple[str, str]:
    """Check if the model is stuck in a repetition loop.

    Returns (verdict, message) where verdict is one of:
      - "ok"    — no issue, proceed normally
      - "nudge" — issue detected, skip this action and give the model feedback
      - "stop"  — too many nudges failed, truly halt the run

    Key principle: if the model changed its approach after a nudge (e.g., switched
    from clicking to keyboard), ALWAYS let it through — even if nudge_count is maxed.
    Only STOP if the model keeps doing the same broken pattern.
    """
    if not getattr(cfg, "STOP_ON_REPEAT", True):
        return OK, ""

    if not history:
        return OK, ""

    max_nudges = getattr(cfg, "MAX_NUDGES", 3)
    new_sig = action_signature(new_action)
    log.debug("check_repeat: nudge_count=%d, new_action=%s, history_len=%d",
              nudge_count, new_sig, len(history))

    # CRITICAL: If the model changed approach after a nudge, let it through!
    # This prevents killing the model the instant it finally obeys our feedback.
    if nudge_count > 0 and _model_changed_approach(history, new_action):
        return OK, ""

    # --- 1. Oscillation detection ---
    osc, osc_msg = _detect_oscillation(history, new_action, nudge_count)
    if osc:
        if nudge_count >= max_nudges:
            log.warning("STOP (oscillation, %d nudges exhausted): %s", nudge_count, osc_msg)
            return STOP, f"Stopping after {nudge_count} failed correction attempts. {osc_msg}"
        log.warning("NUDGE (oscillation): %s", osc_msg)
        return NUDGE, osc_msg

    # --- 2. No-progress detection ---
    nop, nop_msg = _detect_no_progress(history, new_action, nudge_count)
    if nop:
        if nudge_count >= max_nudges:
            log.warning("STOP (no-progress, %d nudges exhausted): %s", nudge_count, nop_msg)
            return STOP, f"Stopping after {nudge_count} failed correction attempts. {nop_msg}"
        log.warning("NUDGE (no-progress): %s", nop_msg)
        return NUDGE, nop_msg

    # --- 3. Direct repeat checks ---
    rep, rep_msg = _detect_direct_repeat(history, new_action)
    if rep:
        if nudge_count >= max_nudges:
            log.warning("STOP (direct-repeat, %d nudges exhausted): %s", nudge_count, rep_msg)
            return STOP, f"Stopping after {nudge_count} failed correction attempts. {rep_msg}"
        log.warning("NUDGE (direct-repeat): %s", rep_msg)
        return NUDGE, rep_msg

    return OK, ""


# Backward compatibility alias
def should_stop_on_repeat(history: List[Dict[str, Any]], new_action: Dict[str, Any]) -> Tuple[bool, str]:
    """Deprecated — use check_repeat() instead."""
    verdict, msg = check_repeat(history, new_action)
    return verdict == STOP, msg
