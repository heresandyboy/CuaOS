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
    if handler_type == "qwen25vl":
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
# Unified entry point
# ═══════════════════════════════════════════

def ask_next_action(llm: Llama, objective: str, screenshot_path: str, history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Returns one action dict. When done: {"action":"BITTI", ...}
    Dispatches to the correct prompt/parser based on cfg.CHAT_HANDLER.
    """
    uri = image_to_data_uri(screenshot_path)

    if cfg.CHAT_HANDLER == "qwen25vl":
        # UI-TARS — needs image dimensions for coordinate conversion
        from PIL import Image
        with Image.open(screenshot_path) as img:
            img_w, img_h = img.size
        return _ask_uitars(llm, objective, uri, history, img_w, img_h)

    # Default: Qwen3-VL with JSON output
    return _ask_qwen3vl(llm, objective, uri, history)
