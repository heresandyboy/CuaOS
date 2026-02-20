"""Tests for Fara-7B integration — parser, key mapping, system prompt, history.

Mocks llama_cpp since CUDA DLLs aren't available in test environment.
"""
import json
import sys
import os
import types

# Mock llama_cpp before any src imports
mock_llama = types.ModuleType("llama_cpp")
mock_llama.Llama = type("Llama", (), {})
sys.modules["llama_cpp"] = mock_llama

# Mock llama_cpp.llama_chat_format
mock_chat_format = types.ModuleType("llama_cpp.llama_chat_format")
mock_chat_format.Qwen25VLChatHandler = type("Qwen25VLChatHandler", (), {"__init__": lambda self, **kw: None})
mock_chat_format.Qwen3VLChatHandler = type("Qwen3VLChatHandler", (), {"__init__": lambda self, **kw: None})
sys.modules["llama_cpp.llama_chat_format"] = mock_chat_format

# Mock huggingface_hub
mock_hf = types.ModuleType("huggingface_hub")
mock_hf.hf_hub_download = lambda **kw: "/mock/path"
sys.modules["huggingface_hub"] = mock_hf

# Mock PIL (for image_to_data_uri and vision.py)
mock_pil = types.ModuleType("PIL")
mock_pil_image = types.ModuleType("PIL.Image")
mock_pil_image.open = lambda *a, **kw: None
mock_pil_image.Image = type("Image", (), {})
mock_pil_imagedraw = types.ModuleType("PIL.ImageDraw")
mock_pil_imagedraw.Draw = lambda *a, **kw: type("MockDraw", (), {"ellipse": lambda *a, **kw: None})()
mock_pil.Image = mock_pil_image
mock_pil.ImageDraw = mock_pil_imagedraw
sys.modules["PIL"] = mock_pil
sys.modules["PIL.Image"] = mock_pil_image
sys.modules["PIL.ImageDraw"] = mock_pil_imagedraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm_client import (
    _parse_fara_output,
    _build_fara_system_prompt,
    _fara_map_key,
    _strip_old_images,
    _smart_resize,
    _FARA_KEY_MAP,
    _fara_chat_history,
    reset_fara_history,
)
from src.guards import action_signature, check_repeat, _model_changed_approach

# Test image dimensions (1920x1080 screen)
IMG_W = 1920
IMG_H = 1080

# What smart_resize gives us for 1920x1080
SMART_H, SMART_W = _smart_resize(IMG_H, IMG_W)
print(f"smart_resize(1080, 1920) = ({SMART_H}, {SMART_W})")

passed = 0
failed = 0

def check(name, got, expected):
    global passed, failed
    if got == expected:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name}")
        print(f"    expected: {expected}")
        print(f"    got:      {got}")


# ═══════════════════════════════════════════
# 1. Key mapping
# ═══════════════════════════════════════════
print("\n=== Key Mapping ===")
check("Enter -> enter", _fara_map_key("Enter"), "enter")
check("Return -> enter", _fara_map_key("Return"), "enter")
check("ArrowUp -> up", _fara_map_key("ArrowUp"), "up")
check("ArrowDown -> down", _fara_map_key("ArrowDown"), "down")
check("ArrowLeft -> left", _fara_map_key("ArrowLeft"), "left")
check("ArrowRight -> right", _fara_map_key("ArrowRight"), "right")
check("Control -> ctrl", _fara_map_key("Control"), "ctrl")
check("Shift -> shift", _fara_map_key("Shift"), "shift")
check("Alt -> alt", _fara_map_key("Alt"), "alt")
check("Escape -> esc", _fara_map_key("Escape"), "esc")
check("Backspace -> backspace", _fara_map_key("Backspace"), "backspace")
check("Delete -> delete", _fara_map_key("Delete"), "delete")
check("Space -> space", _fara_map_key("Space"), "space")
check("Tab -> tab", _fara_map_key("Tab"), "tab")
check("Home -> home", _fara_map_key("Home"), "home")
check("End -> end", _fara_map_key("End"), "end")
check("PageUp -> pageup", _fara_map_key("PageUp"), "pageup")
check("PageDown -> pagedown", _fara_map_key("PageDown"), "pagedown")
check("Meta -> super", _fara_map_key("Meta"), "super")
check("unknown 'a' -> 'a'", _fara_map_key("a"), "a")
check("F5 -> f5", _fara_map_key("F5"), "f5")
check("F12 -> f12", _fara_map_key("F12"), "f12")


# ═══════════════════════════════════════════
# 2. Parser: left_click
# ═══════════════════════════════════════════
print("\n=== Parser: left_click ===")
text = """I see the search button in the top right area. I'll click it.
<tool_call>
{"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [714, 448]}}
</tool_call>"""

result = _parse_fara_output(text, IMG_W, IMG_H)
check("action", result["action"], "CLICK")
check("x normalized", round(result["x"], 4), round(714 / SMART_W, 4))
check("y normalized", round(result["y"], 4), round(448 / SMART_H, 4))
check("has thought", "search button" in result.get("target", ""), True)


# ═══════════════════════════════════════════
# 3. Parser: visit_url
# ═══════════════════════════════════════════
print("\n=== Parser: visit_url ===")
text = """I need to navigate to the HuggingFace models page.
<tool_call>
{"name": "computer_use", "arguments": {"action": "visit_url", "url": "https://huggingface.co/models"}}
</tool_call>"""

result = _parse_fara_output(text, IMG_W, IMG_H)
check("action", result["action"], "VISIT_URL")
check("url", result["url"], "https://huggingface.co/models")
check("has thought", "HuggingFace" in result.get("target", ""), True)


# ═══════════════════════════════════════════
# 4. Parser: web_search
# ═══════════════════════════════════════════
print("\n=== Parser: web_search ===")
text = """I'll search for the model.
<tool_call>
{"name": "computer_use", "arguments": {"action": "web_search", "query": "Fara-7B model download"}}
</tool_call>"""

result = _parse_fara_output(text, IMG_W, IMG_H)
check("action", result["action"], "WEB_SEARCH")
check("query", result["query"], "Fara-7B model download")


# ═══════════════════════════════════════════
# 5. Parser: type with coordinate and press_enter
# ═══════════════════════════════════════════
print("\n=== Parser: type with coordinate ===")
text = """I'll type in the search box.
<tool_call>
{"name": "computer_use", "arguments": {"action": "type", "text": "hello world", "coordinate": [500, 300], "press_enter": false, "delete_existing_text": true}}
</tool_call>"""

result = _parse_fara_output(text, IMG_W, IMG_H)
check("action", result["action"], "TYPE")
check("text", result["text"], "hello world")
check("has click_x", "click_x" in result, True)
check("click_x normalized", round(result["click_x"], 4), round(500 / SMART_W, 4))
check("click_y normalized", round(result["click_y"], 4), round(300 / SMART_H, 4))
check("press_enter false", result["press_enter"], False)
check("delete_existing true", result["delete_existing"], True)


# ═══════════════════════════════════════════
# 6. Parser: type without coordinate (just text)
# ═══════════════════════════════════════════
print("\n=== Parser: type without coordinate ===")
text = """Type the URL.
<tool_call>
{"name": "computer_use", "arguments": {"action": "type", "text": "https://example.com"}}
</tool_call>"""

result = _parse_fara_output(text, IMG_W, IMG_H)
check("action", result["action"], "TYPE")
check("text", result["text"], "https://example.com")
check("no click_x", "click_x" not in result, True)
check("press_enter default true", result["press_enter"], True)
check("delete_existing default false", result["delete_existing"], False)


# ═══════════════════════════════════════════
# 7. Parser: key (single)
# ═══════════════════════════════════════════
print("\n=== Parser: key (single) ===")
text = """Press Enter to submit.
<tool_call>
{"name": "computer_use", "arguments": {"action": "key", "keys": ["Enter"]}}
</tool_call>"""

result = _parse_fara_output(text, IMG_W, IMG_H)
check("action", result["action"], "PRESS")
check("key mapped", result["key"], "enter")


# ═══════════════════════════════════════════
# 8. Parser: key (combo)
# ═══════════════════════════════════════════
print("\n=== Parser: key (combo) ===")
text = """Select all with Ctrl+A.
<tool_call>
{"name": "computer_use", "arguments": {"action": "key", "keys": ["Control", "a"]}}
</tool_call>"""

result = _parse_fara_output(text, IMG_W, IMG_H)
check("action", result["action"], "HOTKEY")
check("keys mapped", result["keys"], ["ctrl", "a"])


# ═══════════════════════════════════════════
# 9. Parser: key (triple combo)
# ═══════════════════════════════════════════
print("\n=== Parser: key (triple combo) ===")
text = """Open task manager.
<tool_call>
{"name": "computer_use", "arguments": {"action": "key", "keys": ["Control", "Shift", "Escape"]}}
</tool_call>"""

result = _parse_fara_output(text, IMG_W, IMG_H)
check("action", result["action"], "HOTKEY")
check("keys mapped", result["keys"], ["ctrl", "shift", "esc"])


# ═══════════════════════════════════════════
# 10. Parser: scroll down
# ═══════════════════════════════════════════
print("\n=== Parser: scroll ===")
text = """Scroll down to see more.
<tool_call>
{"name": "computer_use", "arguments": {"action": "scroll", "pixels": -300}}
</tool_call>"""

result = _parse_fara_output(text, IMG_W, IMG_H)
check("action", result["action"], "SCROLL")
check("scroll down", result["scroll"], -3)

text2 = """Scroll up.
<tool_call>
{"name": "computer_use", "arguments": {"action": "scroll", "pixels": 300}}
</tool_call>"""
result2 = _parse_fara_output(text2, IMG_W, IMG_H)
check("scroll up", result2["scroll"], 3)


# ═══════════════════════════════════════════
# 11. Parser: terminate
# ═══════════════════════════════════════════
print("\n=== Parser: terminate ===")
text = """Task completed successfully.
<tool_call>
{"name": "computer_use", "arguments": {"action": "terminate", "status": "success"}}
</tool_call>"""

result = _parse_fara_output(text, IMG_W, IMG_H)
check("action", result["action"], "BITTI")
check("has status in target", "success" in result.get("target", ""), True)


# ═══════════════════════════════════════════
# 12. Parser: terminate failure
# ═══════════════════════════════════════════
print("\n=== Parser: terminate failure ===")
text = """Cannot complete task.
<tool_call>
{"name": "computer_use", "arguments": {"action": "terminate", "status": "failure"}}
</tool_call>"""

result = _parse_fara_output(text, IMG_W, IMG_H)
check("action", result["action"], "BITTI")
check("has failure in target", "failure" in result.get("target", ""), True)


# ═══════════════════════════════════════════
# 13. Parser: history_back
# ═══════════════════════════════════════════
print("\n=== Parser: history_back ===")
text = """Go back.
<tool_call>
{"name": "computer_use", "arguments": {"action": "history_back"}}
</tool_call>"""

result = _parse_fara_output(text, IMG_W, IMG_H)
check("action", result["action"], "HOTKEY")
check("keys", result["keys"], ["alt", "left"])


# ═══════════════════════════════════════════
# 14. Parser: wait
# ═══════════════════════════════════════════
print("\n=== Parser: wait ===")
text = """Wait for page to load.
<tool_call>
{"name": "computer_use", "arguments": {"action": "wait", "time": 5}}
</tool_call>"""

result = _parse_fara_output(text, IMG_W, IMG_H)
check("action", result["action"], "WAIT")
check("seconds", result["seconds"], 5.0)


# ═══════════════════════════════════════════
# 15. Parser: pause_and_memorize_fact
# ═══════════════════════════════════════════
print("\n=== Parser: pause_and_memorize_fact ===")
text = """I see the price is $29.99.
<tool_call>
{"name": "computer_use", "arguments": {"action": "pause_and_memorize_fact", "fact": "Product price is $29.99"}}
</tool_call>"""

result = _parse_fara_output(text, IMG_W, IMG_H)
check("action", result["action"], "NOOP")
check("has memo", "memorized" in result.get("target", ""), True)


# ═══════════════════════════════════════════
# 16. Parser: mouse_move
# ═══════════════════════════════════════════
print("\n=== Parser: mouse_move ===")
text = """Move to the dropdown.
<tool_call>
{"name": "computer_use", "arguments": {"action": "mouse_move", "coordinate": [600, 400]}}
</tool_call>"""

result = _parse_fara_output(text, IMG_W, IMG_H)
check("action", result["action"], "MOVE")
check("x normalized", round(result["x"], 4), round(600 / SMART_W, 4))
check("y normalized", round(result["y"], 4), round(400 / SMART_H, 4))


# ═══════════════════════════════════════════
# 17. Parser: truncated output (no closing tag)
# ═══════════════════════════════════════════
print("\n=== Parser: truncated output ===")
text = """I'll click the button.
<tool_call>
{"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [100, 200]}}"""

result = _parse_fara_output(text, IMG_W, IMG_H)
check("truncated still parses", result["action"], "CLICK")


# ═══════════════════════════════════════════
# 18. Parser: no tool_call at all
# ═══════════════════════════════════════════
print("\n=== Parser: no tool_call ===")
text = "I'm thinking about what to do next but haven't decided."
result = _parse_fara_output(text, IMG_W, IMG_H)
check("no tool_call -> NOOP", result["action"], "NOOP")


# ═══════════════════════════════════════════
# 19. Parser: malformed JSON
# ═══════════════════════════════════════════
print("\n=== Parser: malformed JSON ===")
text = """Click.
<tool_call>
{not valid json at all}
</tool_call>"""
result = _parse_fara_output(text, IMG_W, IMG_H)
check("malformed JSON -> NOOP", result["action"], "NOOP")


# ═══════════════════════════════════════════
# 20. Parser: empty coordinate
# ═══════════════════════════════════════════
print("\n=== Parser: missing coordinate ===")
text = """Click somewhere.
<tool_call>
{"name": "computer_use", "arguments": {"action": "left_click"}}
</tool_call>"""
result = _parse_fara_output(text, IMG_W, IMG_H)
# No coordinate -> should still be NOOP since _norm_coord returns None
check("click without coordinate -> NOOP", result["action"], "NOOP")


# ═══════════════════════════════════════════
# 21. System prompt
# ═══════════════════════════════════════════
print("\n=== System Prompt ===")
prompt = _build_fara_system_prompt(SMART_W, SMART_H)
check("contains <tools>", "<tools>" in prompt, True)
check("contains </tools>", "</tools>" in prompt, True)
check("contains <tool_call>", "<tool_call>" in prompt, True)
check("contains resolution", f"{SMART_W}x{SMART_H}" in prompt, True)
check("contains visit_url", "visit_url" in prompt, True)
check("contains web_search", "web_search" in prompt, True)
check("contains computer_use", "computer_use" in prompt, True)
check("contains terminate", "terminate" in prompt, True)
check("contains left_click", "left_click" in prompt, True)
check("contains mouse_move", "mouse_move" in prompt, True)

# Verify the embedded JSON is valid
import re
tools_match = re.search(r"<tools>\n(.*?)\n</tools>", prompt, re.DOTALL)
check("tools block parseable", tools_match is not None, True)
if tools_match:
    tool_json = json.loads(tools_match.group(1))
    check("tool name", tool_json["name"], "computer_use")
    actions = tool_json["parameters"]["properties"]["action"]["enum"]
    check("11 actions", len(actions), 11)
    check("has terminate", "terminate" in actions, True)
    check("has left_click", "left_click" in actions, True)
    check("has visit_url", "visit_url" in actions, True)
    check("has web_search", "web_search" in actions, True)
    check("has history_back", "history_back" in actions, True)
    check("has pause_and_memorize_fact", "pause_and_memorize_fact" in actions, True)


# ═══════════════════════════════════════════
# 22. Strip old images
# ═══════════════════════════════════════════
print("\n=== Strip Old Images ===")
messages = [
    {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,img1"}},
        {"type": "text", "text": "Turn 1"},
    ]},
    {"role": "assistant", "content": "Response 1"},
    {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,img2"}},
        {"type": "text", "text": "Turn 2"},
    ]},
    {"role": "assistant", "content": "Response 2"},
    {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,img3"}},
        {"type": "text", "text": "Turn 3"},
    ]},
    {"role": "assistant", "content": "Response 3"},
    {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,img4"}},
        {"type": "text", "text": "Turn 4"},
    ]},
]

stripped = _strip_old_images(messages, keep_last=3)

# Count images in stripped messages
img_count = 0
for msg in stripped:
    content = msg.get("content", [])
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                img_count += 1

check("stripped to 3 images", img_count, 3)

# First user message should have text only (image stripped)
first_user = stripped[0]
check("first user has role", first_user["role"], "user")
has_img = False
if isinstance(first_user.get("content"), list):
    has_img = any(
        isinstance(p, dict) and p.get("type") == "image_url"
        for p in first_user["content"]
    )
check("first user image stripped", has_img, False)

# But text should be retained
has_text = False
if isinstance(first_user.get("content"), list):
    has_text = any(
        isinstance(p, dict) and p.get("type") == "text"
        for p in first_user["content"]
    )
check("first user text retained", has_text, True)

# Last 3 users should still have images
for idx in [2, 4, 6]:  # user messages at indices 2, 4, 6
    if idx < len(stripped):
        msg = stripped[idx]
        if isinstance(msg.get("content"), list):
            has = any(isinstance(p, dict) and p.get("type") == "image_url" for p in msg["content"])
            check(f"msg[{idx}] still has image", has, True)


# ═══════════════════════════════════════════
# 23. Strip old images edge case: fewer than N
# ═══════════════════════════════════════════
print("\n=== Strip Old Images: fewer than N ===")
messages_short = [
    {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,img1"}},
        {"type": "text", "text": "Turn 1"},
    ]},
    {"role": "assistant", "content": "Response 1"},
]
stripped_short = _strip_old_images(messages_short, keep_last=3)
check("short list unchanged length", len(stripped_short), len(messages_short))
has_img = any(
    isinstance(p, dict) and p.get("type") == "image_url"
    for p in stripped_short[0].get("content", [])
    if isinstance(stripped_short[0].get("content"), list)
)
check("single image preserved", has_img, True)


# ═══════════════════════════════════════════
# 24. Reset history
# ═══════════════════════════════════════════
print("\n=== Reset History ===")
_fara_chat_history.append({"role": "test"})
check("history not empty", len(_fara_chat_history) > 0, True)
reset_fara_history()
check("history cleared", len(_fara_chat_history), 0)


# ═══════════════════════════════════════════
# 25. Guard: VISIT_URL/WEB_SEARCH signatures
# ═══════════════════════════════════════════
print("\n=== Guard Signatures ===")
check("VISIT_URL sig", action_signature({"action": "VISIT_URL", "url": "https://example.com"}),
      "VISIT_URL:https://example.com")
check("WEB_SEARCH sig", action_signature({"action": "WEB_SEARCH", "query": "test query"}),
      "WEB_SEARCH:test query")


# ═══════════════════════════════════════════
# 26. Guard: approach change detection
# ═══════════════════════════════════════════
print("\n=== Guard: Approach Change ===")
history = [
    {"action": "CLICK", "x": 0.5, "y": 0.5},
    {"action": "CLICK", "x": 0.5, "y": 0.5},
    {"action": "SYSTEM_FEEDBACK", "target": "You are stuck clicking"},
]
new_visit = {"action": "VISIT_URL", "url": "https://example.com"}
new_click = {"action": "CLICK", "x": 0.5, "y": 0.5}
new_search = {"action": "WEB_SEARCH", "query": "test"}
new_type = {"action": "TYPE", "text": "hello"}

check("VISIT_URL is changed approach (click->nav)", _model_changed_approach(history, new_visit), True)
check("WEB_SEARCH is changed approach (click->nav)", _model_changed_approach(history, new_search), True)
check("TYPE is changed approach (click->keyboard)", _model_changed_approach(history, new_type), True)
check("same CLICK is NOT changed", _model_changed_approach(history, new_click), False)


# ═══════════════════════════════════════════
# 27. Smart resize consistency
# ═══════════════════════════════════════════
print("\n=== Smart Resize ===")
h, w = _smart_resize(1080, 1920)
check("height divisible by 28", h % 28, 0)
check("width divisible by 28", w % 28, 0)
check("reasonable height", 500 <= h <= 2000, True)
check("reasonable width", 900 <= w <= 3000, True)

# Test various resolutions
for test_h, test_w in [(720, 1280), (1080, 1920), (2160, 3840), (600, 800)]:
    sh, sw = _smart_resize(test_h, test_w)
    check(f"resize {test_w}x{test_h}: h%28==0", sh % 28, 0)
    check(f"resize {test_w}x{test_h}: w%28==0", sw % 28, 0)


# ═══════════════════════════════════════════
# 28. Coordinate round-trip: parse -> normalize -> valid range
# ═══════════════════════════════════════════
print("\n=== Coordinate Range ===")
# Test corners and center
for px, py, label in [(0, 0, "top-left"), (SMART_W, SMART_H, "bottom-right"),
                       (SMART_W//2, SMART_H//2, "center"),
                       (100, 100, "near-origin"), (SMART_W-1, SMART_H-1, "near-max")]:
    text = f"""Click {label}.
<tool_call>
{{"name": "computer_use", "arguments": {{"action": "left_click", "coordinate": [{px}, {py}]}}}}
</tool_call>"""
    r = _parse_fara_output(text, IMG_W, IMG_H)
    if r["action"] == "CLICK":
        x, y = r["x"], r["y"]
        in_range = 0.0 <= x <= 1.05 and 0.0 <= y <= 1.05  # slight tolerance for edge
        check(f"coord {label} ({px},{py}) in [0,1]: x={x:.4f} y={y:.4f}", in_range, True)


# ═══════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════
print(f"\n{'='*50}")
print(f"Results: {passed} passed, {failed} failed")
if failed > 0:
    print("SOME TESTS FAILED!")
    sys.exit(1)
else:
    print("ALL TESTS PASSED!")
    sys.exit(0)
