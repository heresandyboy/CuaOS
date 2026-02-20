# llm_client.py
from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

from src.config import cfg, JSON_RE
from src.log import get_logger
from src.vision import image_to_data_uri

log = get_logger("llm")


# ═══════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════

def _make_chat_handler(handler_type: str, mmproj_path: str):
    """Return the appropriate vision chat handler for the active model."""
    if handler_type == "qwen3vl":
        from llama_cpp.llama_chat_format import Qwen3VLChatHandler
        return Qwen3VLChatHandler(
            clip_model_path=mmproj_path,
            force_reasoning=cfg.FORCE_REASONING,
            image_min_tokens=cfg.IMAGE_MIN_TOKENS,
        )
    if handler_type in ("qwen25vl", "fara"):
        from llama_cpp.llama_chat_format import Qwen25VLChatHandler
        return Qwen25VLChatHandler(
            clip_model_path=mmproj_path,
            image_min_tokens=cfg.IMAGE_MIN_TOKENS,
            image_max_tokens=cfg.IMAGE_MAX_TOKENS,
        )
    raise ValueError(f"Unknown chat handler type: {handler_type}")


def load_llm() -> Llama:
    log.info("Loading model: %s", cfg.MODEL_NAME)
    log.info("  repo:    %s", cfg.GGUF_REPO_ID)
    log.info("  model:   %s", cfg.GGUF_MODEL_FILENAME)
    log.info("  mmproj:  %s", cfg.GGUF_MMPROJ_FILENAME)
    log.info("  handler: %s", cfg.CHAT_HANDLER)
    log.info("  n_ctx: %d  n_batch: %d", cfg.N_CTX, cfg.N_BATCH)
    log.info("  n_gpu_layers: %s (%s)", cfg.N_GPU_LAYERS,
             "all" if cfg.N_GPU_LAYERS == -1 else cfg.N_GPU_LAYERS)

    model_path = hf_hub_download(repo_id=cfg.GGUF_REPO_ID, filename=cfg.GGUF_MODEL_FILENAME)
    mmproj_path = hf_hub_download(repo_id=cfg.GGUF_REPO_ID, filename=cfg.GGUF_MMPROJ_FILENAME)

    handler = _make_chat_handler(cfg.CHAT_HANDLER, mmproj_path)

    return Llama(
        model_path=model_path,
        chat_handler=handler,
        n_ctx=cfg.N_CTX,
        n_batch=cfg.N_BATCH,
        n_gpu_layers=cfg.N_GPU_LAYERS,
        n_threads=cfg.N_THREADS,
        verbose=False,
    )


# ═══════════════════════════════════════════
# Qwen3-VL prompt & parser  (JSON output)
# ═══════════════════════════════════════════

def _fix_malformed_json(raw: str) -> str:
    """Fix common model JSON issues like 'x': 42, 129, -> 'x': 42, 'y': 129,"""
    raw = re.sub(
        r'"x"\s*:\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,',
        r'"x": \1, "y": \2,',
        raw,
    )
    raw = re.sub(r',\s*}', '}', raw)
    return raw


def _normalize_coords(obj: Dict[str, Any]) -> Dict[str, Any]:
    """If x/y appear to be pixel coords (>1.0), normalize to 0-1 using MAX_DIM."""
    for key in ("x", "y"):
        val = obj.get(key)
        if isinstance(val, (int, float)) and val > 1.0:
            obj[key] = val / cfg.MAX_DIM
    return obj


def _parse_json_obj(text: str) -> Dict[str, Any]:
    m = JSON_RE.search(text.strip())
    if not m:
        raise ValueError(f"Model output is not JSON: {text}")
    raw = m.group(0)
    try:
        return _normalize_coords(json.loads(raw))
    except json.JSONDecodeError:
        fixed = _fix_malformed_json(raw)
        return _normalize_coords(json.loads(fixed))


_QWEN3VL_SYSTEM = (
    "You are a reactive GUI agent controlling a Linux desktop.\n"
    "Given OBJECTIVE, HISTORY, and a SCREENSHOT, decide the NEXT single action.\n\n"
    "Return EXACTLY one JSON object. No extra text.\n"
    "Schema:\n"
    "{\n"
    '  "action": "CLICK|DOUBLE_CLICK|RIGHT_CLICK|TYPE|PRESS|HOTKEY|SCROLL|WAIT|NOOP|BITTI",\n'
    '  "x": 0.5,\n'
    '  "y": 0.5,\n'
    '  "text": "",\n'
    '  "key": "",\n'
    '  "keys": [""],\n'
    '  "scroll": 0,\n'
    '  "seconds": 0.0,\n'
    '  "target": "short description",\n'
    '  "confidence": 0.0,\n'
    '  "why_short": "<=12 words"\n'
    "}\n\n"
    "CRITICAL RULES:\n"
    "- Output ONLY valid JSON.\n"
    "- For CLICK/DOUBLE_CLICK/RIGHT_CLICK: set x,y (normalized 0.0 to 1.0).\n"
    "- For TYPE: set text.\n"
    "- For PRESS: set key.\n"
    "- For HOTKEY: set keys list.\n"
    "- For SCROLL: set scroll (positive=up, negative=down).\n"
    "- For WAIT: set seconds.\n"
    "- If objective is complete, action MUST be BITTI.\n"
    "- NEVER repeat a failed action. If an action had no effect (marked ❌), do something DIFFERENT.\n"
    "- If clicking is not working, switch to keyboard: HOTKEY keys=['ctrl','l'] to focus address bar, then TYPE.\n"
    "- Pay close attention to ⚠️ WARNING messages in your history.\n"
    f"- Safety: Never output x or y within {cfg.MIN_MARGIN} of edges.\n"
)


def _format_qwen3vl_history(history: List[Dict[str, Any]]) -> str:
    """Convert internal history to human-readable text for Qwen3-VL.

    Marks failed actions with ❌ and prominently displays SYSTEM_FEEDBACK warnings.
    """
    if not history:
        return ""
    lines = []
    for i, h in enumerate(history, 1):
        act = h.get("action", "?")

        if act == "SYSTEM_FEEDBACK":
            feedback = h.get("target", "")
            lines.append(f"Step {i}: ⚠️ WARNING: {feedback}")
            continue

        # Build a short description of what was done
        desc_parts = [act]
        if act in ("CLICK", "DOUBLE_CLICK", "RIGHT_CLICK"):
            hx, hy = h.get("x"), h.get("y")
            if isinstance(hx, (int, float)) and isinstance(hy, (int, float)):
                desc_parts.append(f"at ({hx:.4f}, {hy:.4f})")
        elif act == "TYPE":
            text = h.get("text", "")
            desc_parts.append(f"'{text[:40]}'" if text else "")
        elif act == "PRESS":
            desc_parts.append(h.get("key", ""))
        elif act == "HOTKEY":
            desc_parts.append("+".join(h.get("keys") or []))
        elif act == "SCROLL":
            desc_parts.append(str(h.get("scroll", 0)))

        target = h.get("target", h.get("why_short", ""))
        if target:
            if len(target) > 60:
                target = target[:57] + "..."
            desc_parts.append(f"— {target}")

        sc = h.get("screen_changed")
        if sc is False:
            desc_parts.append("❌ NO EFFECT")
        elif sc is True:
            desc_parts.append("✓")

        lines.append(f"Step {i}: {' '.join(p for p in desc_parts if p)}")
    return "\n".join(lines)


def _build_qwen3vl_instruction(objective: str, history: List[Dict[str, Any]]) -> str:
    """Build the full user prompt for Qwen3-VL, with prominent feedback warnings."""
    parts = [f"OBJECTIVE: {objective}"]

    # If the last action had no visible effect or guard sent feedback, add LOUD warning
    if history:
        last = history[-1]
        last_action = (last.get("action") or "").upper()

        if last_action == "SYSTEM_FEEDBACK":
            feedback = last.get("target", "")
            parts.append(
                f"\n⚠️ CRITICAL WARNING: {feedback}\n"
                "You MUST try a COMPLETELY DIFFERENT approach. "
                "Do NOT click the same area. Use keyboard instead:\n"
                '- HOTKEY keys=["ctrl","l"] to focus address bar, then TYPE to enter URL\n'
                '- HOTKEY keys=["ctrl","t"] to open new tab\n'
                "- SCROLL to find different elements\n"
                "- Click a DIFFERENT part of the screen"
            )
        elif last.get("screen_changed") is False:
            parts.append(
                "\n⚠️ WARNING: Your last action had NO visible effect on the screen. "
                "That click/action did NOT work. Try something DIFFERENT."
            )

    # Add formatted history
    history_text = _format_qwen3vl_history(history)
    if history_text:
        parts.append(f"\nHISTORY:\n{history_text}")
    else:
        parts.append("\nHISTORY: (none)")

    parts.append("\nDecide the NEXT action from the CURRENT screenshot. Output ONLY JSON.")

    return "\n".join(parts)


def _ask_qwen3vl(llm: Llama, objective: str, uri: str, history: List[Dict[str, Any]]) -> Dict[str, Any]:
    user_text = _build_qwen3vl_instruction(objective, history)
    log.debug("Qwen3-VL prompt:\n%s", user_text)

    resp = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": _QWEN3VL_SYSTEM},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": uri}},
                {"type": "text", "text": user_text},
            ]},
        ],
        temperature=0.1,
        top_p=0.9,
        max_tokens=220,
        stop=["\n\n", "<|im_end|>"],
    )
    return _parse_json_obj(resp["choices"][0]["message"]["content"])


# ═══════════════════════════════════════════
# UI-TARS prompt & parser  (Thought/Action output)
# ═══════════════════════════════════════════

_UITARS_SYSTEM = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
```
Thought: ...
Action: ...
```

## Action Space
click(start_box='(x1,y1)')
left_double(start_box='(x1,y1)')
right_single(start_box='(x1,y1)')
drag(start_box='(x1,y1)', end_box='(x2,y2)')
hotkey(key='ctrl c')
type(content='xxx')
scroll(start_box='(x1,y1)', direction='down or up or right or left')
wait()
finished(content='xxx')

## Note
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
"""


def _smart_resize(height: int, width: int) -> Tuple[int, int]:
    """Compute the dimensions Qwen2.5-VL uses internally (matches C++ clip.cpp).

    Coordinates the model outputs are relative to these dimensions.
    Must use the SAME min/max pixels as the C++ clip model to get correct
    coordinate conversion.  We derive them from cfg.IMAGE_MIN/MAX_TOKENS.
    """
    factor = 28
    # Must match the values the C++ clip model uses (token * 28^2)
    min_pixels = cfg.IMAGE_MIN_TOKENS * factor * factor   # 1024*784 = 802_816
    max_pixels = cfg.IMAGE_MAX_TOKENS * factor * factor   # 4096*784 = 3_211_264

    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return h_bar, w_bar


# Regex to extract action calls like: click(start_box='(235,512)')
_UITARS_ACTION_RE = re.compile(
    r"Action:\s*(.+?)(?:\n|$)", re.IGNORECASE
)
_UITARS_COORD_RE = re.compile(
    r"\((\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\)"
)
_UITARS_THOUGHT_RE = re.compile(
    r"Thought:\s*(.+?)(?:\nAction:|\n\n|$)", re.DOTALL
)


def _parse_uitars_output(text: str, img_w: int, img_h: int) -> Dict[str, Any]:
    """Parse UI-TARS Thought/Action output into our internal action dict."""
    text = text.strip()

    # Extract thought
    thought_m = _UITARS_THOUGHT_RE.search(text)
    thought = thought_m.group(1).strip() if thought_m else ""

    # Extract action line
    action_m = _UITARS_ACTION_RE.search(text)
    if not action_m:
        return {"action": "NOOP", "why_short": f"Could not parse: {text[:80]}"}

    action_str = action_m.group(1).strip()

    # Compute smart_resize dimensions for coordinate conversion
    smart_h, smart_w = _smart_resize(img_h, img_w)

    # Parse the action type and arguments
    action_lower = action_str.lower()

    if action_lower.startswith("click("):
        coords = _UITARS_COORD_RE.search(action_str)
        if coords:
            raw_x, raw_y = float(coords.group(1)), float(coords.group(2))
            x = raw_x / smart_w
            y = raw_y / smart_h
            log.info("CLICK (%s,%s) / (%s,%s) -> norm (%.4f,%.4f)", raw_x, raw_y, smart_w, smart_h, x, y)
            return {"action": "CLICK", "x": x, "y": y,
                    "target": thought, "why_short": thought}

    elif action_lower.startswith("left_double("):
        coords = _UITARS_COORD_RE.search(action_str)
        if coords:
            x = float(coords.group(1)) / smart_w
            y = float(coords.group(2)) / smart_h
            return {"action": "DOUBLE_CLICK", "x": x, "y": y,
                    "target": thought, "why_short": thought}

    elif action_lower.startswith("right_single("):
        coords = _UITARS_COORD_RE.search(action_str)
        if coords:
            x = float(coords.group(1)) / smart_w
            y = float(coords.group(2)) / smart_h
            return {"action": "RIGHT_CLICK", "x": x, "y": y,
                    "target": thought, "why_short": thought}

    elif action_lower.startswith("type("):
        content_m = re.search(r"content='(.*?)'", action_str, re.DOTALL)
        if not content_m:
            content_m = re.search(r'content="(.*?)"', action_str, re.DOTALL)
        text_val = content_m.group(1) if content_m else ""
        # Unescape
        text_val = text_val.replace("\\n", "\n").replace("\\'", "'").replace('\\"', '"')
        return {"action": "TYPE", "text": text_val,
                "target": thought, "why_short": thought}

    elif action_lower.startswith("hotkey("):
        key_m = re.search(r"key='(.*?)'", action_str)
        if not key_m:
            key_m = re.search(r'key="(.*?)"', action_str)
        keys_str = key_m.group(1) if key_m else ""
        keys = keys_str.split()
        if len(keys) == 1:
            return {"action": "PRESS", "key": keys[0],
                    "target": thought, "why_short": thought}
        return {"action": "HOTKEY", "keys": keys,
                "target": thought, "why_short": thought}

    elif action_lower.startswith("scroll("):
        dir_m = re.search(r"direction='(.*?)'", action_str)
        if not dir_m:
            dir_m = re.search(r'direction="(.*?)"', action_str)
        direction = (dir_m.group(1) if dir_m else "down").lower()
        scroll_val = -3 if direction in ("down", "right") else 3
        return {"action": "SCROLL", "scroll": scroll_val,
                "target": thought, "why_short": thought}

    elif action_lower.startswith("wait("):
        return {"action": "WAIT", "seconds": 5.0,
                "target": thought, "why_short": thought}

    elif action_lower.startswith("finished("):
        return {"action": "BITTI",
                "target": thought, "why_short": thought}

    # Fallback
    return {"action": "NOOP", "why_short": f"Unknown UI-TARS action: {action_str[:60]}"}


def _format_uitars_history(history: List[Dict[str, Any]]) -> str:
    """Convert internal history to a brief text summary for UI-TARS context.

    Keeps descriptions short to save tokens. Prominently marks failed actions.
    """
    if not history:
        return ""
    lines = []
    for i, h in enumerate(history, 1):
        act = h.get("action", "?")

        if act == "SYSTEM_FEEDBACK":
            feedback = h.get("target", "")
            lines.append(f"Step {i}: ⚠️ FEEDBACK: {feedback}")
            continue

        # Keep target descriptions short (first sentence only)
        target = h.get("target", h.get("why_short", ""))
        if target and len(target) > 80:
            # Truncate to first sentence or 80 chars
            dot = target.find(". ")
            if 0 < dot < 100:
                target = target[:dot + 1]
            else:
                target = target[:77] + "..."

        sc = h.get("screen_changed")
        if sc is False:
            line = f"Step {i}: {act} — {target} ❌ FAILED (no screen change)"
        elif sc is True:
            line = f"Step {i}: {act} — {target} ✓"
        else:
            line = f"Step {i}: {act} — {target}"
        lines.append(line)
    return "\n".join(lines)


def _build_uitars_instruction(objective: str, history: List[Dict[str, Any]]) -> str:
    """Build the full instruction text for UI-TARS, with prominent feedback."""
    parts = [objective]

    # If the last action had no visible effect, put a LOUD warning at the top
    if history:
        last = history[-1]
        last_action = (last.get("action") or "").upper()

        if last_action == "SYSTEM_FEEDBACK":
            # Guard feedback — make it impossible to miss
            feedback = last.get("target", "")
            parts.append(
                f"\n\n⚠️ CRITICAL WARNING: {feedback}\n"
                "You MUST try a fundamentally different approach. "
                "Do NOT click the same area again. Consider:\n"
                "- Using keyboard shortcuts (hotkey) instead of clicking\n"
                "- Typing a URL directly in the address bar\n"
                "- Scrolling to find different elements\n"
                "- Clicking a completely different part of the screen"
            )
        elif last.get("screen_changed") is False:
            parts.append(
                "\n\n⚠️ WARNING: Your last action had NO visible effect on the screen. "
                "The click/action did not work. Try a different element or approach."
            )

    # Add condensed history
    history_text = _format_uitars_history(history)
    if history_text:
        parts.append(f"\n\nPrevious actions:\n{history_text}")

    return "\n".join(parts)


def _ask_uitars(llm: Llama, objective: str, uri: str, history: List[Dict[str, Any]],
                img_w: int, img_h: int) -> Dict[str, Any]:
    instruction = _build_uitars_instruction(objective, history)

    smart_h, smart_w = _smart_resize(img_h, img_w)
    log.info("Image %dx%d -> smart_resize %dx%d (%d tokens)",
             img_w, img_h, smart_w, smart_h, (smart_h // 28) * (smart_w // 28))

    resp = llm.create_chat_completion(
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": _UITARS_SYSTEM + instruction},
            ]},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": uri}},
            ]},
        ],
        temperature=0.0,
        top_p=0.9,
        frequency_penalty=1.0,
        max_tokens=1000,
        stop=["<|im_end|>"],
    )
    raw_output = resp["choices"][0]["message"]["content"]
    finish = resp["choices"][0].get("finish_reason", "?")
    log.debug("Raw output (%s): %r", finish, raw_output)
    return _parse_uitars_output(raw_output, img_w, img_h)


# ═══════════════════════════════════════════
# Fara-7B prompt & parser  (<tool_call> output)
# ═══════════════════════════════════════════

# Key name mapping: Fara emits DOM-style names, CuaOS sandbox expects xdotool names
_FARA_KEY_MAP: Dict[str, str] = {
    "Enter": "enter", "Return": "enter",
    "Tab": "tab",
    "Escape": "esc",
    "Backspace": "backspace",
    "Delete": "delete",
    "ArrowUp": "up", "ArrowDown": "down",
    "ArrowLeft": "left", "ArrowRight": "right",
    "Home": "home", "End": "end",
    "PageUp": "pageup", "PageDown": "pagedown",
    "Control": "ctrl", "Shift": "shift", "Alt": "alt",
    "Meta": "super", "Space": "space",
    "F1": "f1", "F2": "f2", "F3": "f3", "F4": "f4", "F5": "f5",
    "F6": "f6", "F7": "f7", "F8": "f8", "F9": "f9", "F10": "f10",
    "F11": "f11", "F12": "f12",
}


def _fara_map_key(key: str) -> str:
    """Map a Fara key name to the xdotool name used by the sandbox."""
    return _FARA_KEY_MAP.get(key, key.lower())


def _build_fara_system_prompt(screen_w: int, screen_h: int) -> str:
    """Build the Fara-7B system prompt with dynamic screen resolution.

    Follows the exact format from Microsoft's fara/_prompts.py: a description of
    the computer_use tool with the <tools> block, plus the fn_call template.
    """
    tool_def = json.dumps({
        "name": "computer_use",
        "description": (
            "Use a mouse and keyboard to interact with a computer, and take screenshots.\n"
            "* This is a Linux XFCE desktop. You can see the desktop with icons and a taskbar.\n"
            "* IMPORTANT: If no browser window is visible on screen, you MUST first open Firefox "
            "by double-clicking its icon on the desktop or in the taskbar. Only after Firefox is "
            "open and visible can you use visit_url or web_search actions.\n"
            "* visit_url and web_search REQUIRE a browser to be open. They use keyboard shortcuts "
            "(Ctrl+L) that only work inside a browser window. If no browser is visible, these "
            "actions will fail silently. Always check the screenshot first.\n"
            "* You do not have access to a terminal or applications menu. You must click on "
            "desktop icons to start applications.\n"
            "* Some applications may take time to start or process actions, so you may need to "
            "wait and take successive screenshots to see the results of your actions. E.g. if "
            "you click on Firefox and a window doesn't open, try wait and taking another screenshot.\n"
            f"* The screen's resolution is {screen_w}x{screen_h}.\n"
            "* Whenever you intend to move the cursor to click on an element like an icon, "
            "you should consult a screenshot to determine the coordinates of the element "
            "before moving the cursor.\n"
            "* If you tried clicking on a program or link but it failed to load, even after "
            "waiting, try adjusting your cursor position so that the tip of the cursor visually "
            "falls on the element that you want to click.\n"
            "* Make sure to click any buttons, links, icons, etc with the cursor tip in the center "
            "of the element. Don't click boxes on their edges unless asked.\n"
            "* When a separate scrollable container prominently overlays the webpage, if you want "
            "to scroll within it, you typically need to mouse_move() over it first and then scroll().\n"
            "* If a popup window appears that you want to close, if left_click() on the 'X' or close "
            "button doesn't work, try key(keys=['Escape']) to close it.\n"
            "* On some search bars, when you type(), you may need to press_enter=False and instead "
            "separately call left_click() on the search button to submit the search query.\n"
            "* For calendar widgets, you usually need to left_click() on arrows to move between "
            "months and left_click() on dates to select them; type() is not typically used.\n"
            "* ALWAYS look at the screenshot carefully before choosing an action. Describe what "
            "you see on screen in your thinking before acting."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "key", "type", "mouse_move", "left_click", "scroll",
                        "visit_url", "web_search", "history_back",
                        "pause_and_memorize_fact", "wait", "terminate",
                    ],
                },
                "coordinate": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "The x,y pixel coordinate on the screen.",
                },
                "keys": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Key names for key action.",
                },
                "text": {"type": "string", "description": "Text to type."},
                "url": {"type": "string", "description": "URL for visit_url."},
                "query": {"type": "string", "description": "Query for web_search."},
                "pixels": {"type": "integer", "description": "Scroll amount (positive=up, negative=down)."},
                "time": {"type": "integer", "description": "Wait duration in seconds."},
                "status": {"type": "string", "description": "Termination status (success/failure)."},
                "fact": {"type": "string", "description": "Fact to memorize."},
            },
            "required": ["action"],
        },
    }, indent=2)

    return (
        "You are a helpful assistant.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        "<tools>\n"
        f"{tool_def}\n"
        "</tools>\n\n"
        "For each function call, return a JSON object with function name and arguments "
        "within <tool_call></tool_call> XML tags:\n"
        "<tool_call>\n"
        '{"name": <function-name>, "arguments": <args-json-object>}\n'
        "</tool_call>"
    )


def _parse_fara_output(text: str, img_w: int, img_h: int) -> Dict[str, Any]:
    """Parse Fara-7B <tool_call> output into CuaOS internal action dict."""
    text = text.strip()

    # Extract thought (everything before <tool_call>)
    parts = text.split("<tool_call>")
    thought = parts[0].strip() if len(parts) > 1 else ""

    # Extract JSON from between <tool_call> and </tool_call>
    tc_match = re.search(r"<tool_call>\s*\n?(.*?)\s*\n?</tool_call>", text, re.DOTALL)
    if not tc_match:
        # Try without closing tag (model may have been cut off)
        tc_match = re.search(r"<tool_call>\s*\n?(.*)", text, re.DOTALL)
        if not tc_match:
            return {"action": "NOOP", "why_short": f"No <tool_call> found: {text[:80]}"}

    action_text = tc_match.group(1).strip()
    try:
        action = json.loads(action_text)
    except json.JSONDecodeError:
        try:
            import ast
            action = ast.literal_eval(action_text)
        except Exception:
            return {"action": "NOOP", "why_short": f"Invalid JSON in tool_call: {action_text[:80]}"}

    args = action.get("arguments", {})
    fara_action = args.get("action", "")

    # Compute smart_resize dimensions for coordinate conversion
    smart_h, smart_w = _smart_resize(img_h, img_w)

    # Helper to normalize pixel coords from smart_resize space to 0-1
    def _norm_coord(coord):
        if isinstance(coord, (list, tuple)) and len(coord) >= 2:
            x = float(coord[0]) / smart_w
            y = float(coord[1]) / smart_h
            return x, y
        return None

    if fara_action == "left_click":
        coords = _norm_coord(args.get("coordinate"))
        if coords:
            log.info("FARA CLICK %s / (%s,%s) -> norm (%.4f,%.4f)",
                     args["coordinate"], smart_w, smart_h, coords[0], coords[1])
            return {"action": "CLICK", "x": coords[0], "y": coords[1],
                    "target": thought, "why_short": thought[:80]}

    elif fara_action == "mouse_move":
        coords = _norm_coord(args.get("coordinate"))
        if coords:
            return {"action": "MOVE", "x": coords[0], "y": coords[1],
                    "target": thought, "why_short": thought[:80]}

    elif fara_action == "type":
        text_val = str(args.get("text", ""))
        result = {"action": "TYPE", "text": text_val,
                  "target": thought, "why_short": thought[:80]}
        # If coordinate provided, click there first (handled as compound in actions.py)
        coords = _norm_coord(args.get("coordinate"))
        if coords:
            result["click_x"] = coords[0]
            result["click_y"] = coords[1]
        result["press_enter"] = args.get("press_enter", True)
        result["delete_existing"] = args.get("delete_existing_text", False)
        return result

    elif fara_action == "key":
        keys_raw = args.get("keys", [])
        keys = [_fara_map_key(k) for k in keys_raw]
        if len(keys) == 1:
            return {"action": "PRESS", "key": keys[0],
                    "target": thought, "why_short": thought[:80]}
        return {"action": "HOTKEY", "keys": keys,
                "target": thought, "why_short": thought[:80]}

    elif fara_action == "scroll":
        pixels = int(args.get("pixels", 0))
        # Fara: positive=up, negative=down. CuaOS: positive=up, negative=down. Same!
        scroll_val = 3 if pixels > 0 else -3
        return {"action": "SCROLL", "scroll": scroll_val,
                "target": thought, "why_short": thought[:80]}

    elif fara_action == "visit_url":
        url = str(args.get("url", ""))
        return {"action": "VISIT_URL", "url": url,
                "target": thought, "why_short": f"visit {url[:60]}"}

    elif fara_action == "web_search":
        query = str(args.get("query", ""))
        return {"action": "WEB_SEARCH", "query": query,
                "target": thought, "why_short": f"search: {query[:60]}"}

    elif fara_action == "history_back":
        return {"action": "HOTKEY", "keys": ["alt", "left"],
                "target": thought, "why_short": "browser back"}

    elif fara_action == "wait":
        secs = float(args.get("time", 3))
        return {"action": "WAIT", "seconds": secs,
                "target": thought, "why_short": thought[:80]}

    elif fara_action == "terminate":
        status = args.get("status", "success")
        return {"action": "BITTI",
                "target": f"{status}: {thought}", "why_short": thought[:80]}

    elif fara_action == "pause_and_memorize_fact":
        fact = str(args.get("fact", ""))
        log.info("Fara memorized fact: %s", fact)
        return {"action": "NOOP", "target": f"memorized: {fact}",
                "why_short": f"memo: {fact[:60]}"}

    return {"action": "NOOP", "why_short": f"Unknown Fara action: {fara_action}"}


# Persistent multi-turn history for Fara (kept across calls within one run)
_fara_chat_history: List[Dict[str, Any]] = []
_FARA_MAX_SCREENSHOTS = 3


def reset_fara_history() -> None:
    """Reset Fara multi-turn history (call at start of each new task run)."""
    _fara_chat_history.clear()


def _strip_old_images(messages: List[Dict[str, Any]], keep_last: int = _FARA_MAX_SCREENSHOTS) -> List[Dict[str, Any]]:
    """Return a copy of messages with images removed from all but the last N user messages."""
    # Find indices of user messages that contain images
    img_indices = []
    for i, msg in enumerate(messages):
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        img_indices.append(i)
                        break

    # Keep only the last N images
    to_strip = img_indices[:-keep_last] if len(img_indices) > keep_last else []

    result = []
    for i, msg in enumerate(messages):
        if i in to_strip:
            # Remove image parts, keep text
            content = msg.get("content", [])
            if isinstance(content, list):
                text_parts = [p for p in content if not (isinstance(p, dict) and p.get("type") == "image_url")]
                if text_parts:
                    result.append({"role": msg["role"], "content": text_parts})
                # Skip entirely if only had an image
            else:
                result.append(msg)
        else:
            result.append(msg)
    return result


def _ask_fara(llm: Llama, objective: str, uri: str, history: List[Dict[str, Any]],
              img_w: int, img_h: int) -> Dict[str, Any]:
    """Call Fara-7B with multi-turn conversation history."""
    smart_h, smart_w = _smart_resize(img_h, img_w)
    log.info("Image %dx%d -> smart_resize %dx%d (%d tokens)",
             img_w, img_h, smart_w, smart_h, (smart_h // 28) * (smart_w // 28))

    # Build system prompt with current screenshot's smart_resize dimensions
    system_prompt = _build_fara_system_prompt(smart_w, smart_h)

    # Build user message for this turn
    is_first = len(_fara_chat_history) == 0
    if is_first:
        user_text = (
            f"Task: {objective}\n"
            "Look at the screenshot carefully. Describe what you see on screen, "
            "then decide your first action. If no browser is open and the task requires "
            "web navigation, you must first open Firefox by clicking its icon."
        )
    else:
        # Build observation text from last action result
        obs_parts = []
        if history:
            last = history[-1]
            last_action = (last.get("action") or "").upper()
            if last_action == "SYSTEM_FEEDBACK":
                obs_parts.append(f"IMPORTANT WARNING: {last.get('target', '')}")
            elif last.get("screen_changed") is False:
                obs_parts.append(
                    "Your last action had NO visible effect on the screen. "
                    "The action did not work. You need to try a different approach."
                )
                # Give specific guidance for common failures
                if last_action in ("VISIT_URL", "WEB_SEARCH"):
                    obs_parts.append(
                        "REASON: visit_url/web_search failed because no browser window is active. "
                        "You MUST open Firefox first by clicking its icon on the desktop or taskbar."
                    )
            elif last.get("screen_changed") is True:
                obs_parts.append("The screen has changed after your last action. Good progress.")
        obs_parts.append(f"Reminder — your task: {objective}")
        obs_parts.append("Look at the screenshot and think about what to do next.")
        user_text = "\n".join(obs_parts)

    user_msg = {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": uri}},
        {"type": "text", "text": user_text},
    ]}
    _fara_chat_history.append(user_msg)

    # Strip old images and prepend system message
    convo = _strip_old_images(_fara_chat_history, _FARA_MAX_SCREENSHOTS)
    messages = [{"role": "system", "content": system_prompt}] + convo

    log.debug("Fara conversation: %d messages (%d in history)",
              len(messages), len(_fara_chat_history))

    resp = llm.create_chat_completion(
        messages=messages,
        temperature=0.0,
        max_tokens=1024,
        stop=["<|im_end|>"],
    )
    raw_output = resp["choices"][0]["message"]["content"]
    finish = resp["choices"][0].get("finish_reason", "?")
    log.debug("Fara raw output (%s): %r", finish, raw_output)

    # Add assistant response to history
    _fara_chat_history.append({"role": "assistant", "content": raw_output})

    return _parse_fara_output(raw_output, img_w, img_h)


# ═══════════════════════════════════════════
# Unified entry point
# ═══════════════════════════════════════════

def ask_next_action(llm: Llama, objective: str, screenshot_path: str, history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Returns one action dict. When done: {"action":"BITTI", ...}
    Dispatches to the correct prompt/parser based on cfg.CHAT_HANDLER.
    """
    uri = image_to_data_uri(screenshot_path)

    if cfg.CHAT_HANDLER == "fara":
        from PIL import Image
        with Image.open(screenshot_path) as img:
            img_w, img_h = img.size
        return _ask_fara(llm, objective, uri, history, img_w, img_h)

    if cfg.CHAT_HANDLER == "qwen25vl":
        # UI-TARS — needs image dimensions for coordinate conversion
        from PIL import Image
        with Image.open(screenshot_path) as img:
            img_w, img_h = img.size
        return _ask_uitars(llm, objective, uri, history, img_w, img_h)

    # Default: Qwen3-VL with JSON output
    return _ask_qwen3vl(llm, objective, uri, history)
