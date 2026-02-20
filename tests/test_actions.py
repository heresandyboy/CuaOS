"""Tests for execute_action — especially Fara compound actions and enhanced TYPE."""
import sys
import os
import types
import time

# Mock llama_cpp (needed by config -> llm_client chain)
mock_llama = types.ModuleType("llama_cpp")
mock_llama.Llama = type("Llama", (), {})
sys.modules["llama_cpp"] = mock_llama
mock_chat_format = types.ModuleType("llama_cpp.llama_chat_format")
sys.modules["llama_cpp.llama_chat_format"] = mock_chat_format
mock_hf = types.ModuleType("huggingface_hub")
mock_hf.hf_hub_download = lambda **kw: "/mock/path"
sys.modules["huggingface_hub"] = mock_hf
mock_pil = types.ModuleType("PIL")
mock_pil_image = types.ModuleType("PIL.Image")
mock_pil_image.Image = type("Image", (), {})  # PIL.Image.Image class
mock_pil_image.open = lambda *a, **kw: None
mock_pil_imagedraw = types.ModuleType("PIL.ImageDraw")
mock_pil_imagedraw.Draw = lambda *a, **kw: type("MockDraw", (), {"ellipse": lambda *a, **kw: None})()
mock_pil.Image = mock_pil_image
mock_pil.ImageDraw = mock_pil_imagedraw
sys.modules["PIL"] = mock_pil
sys.modules["PIL.Image"] = mock_pil_image
sys.modules["PIL.ImageDraw"] = mock_pil_imagedraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock Sandbox
class MockSandbox:
    """Records all calls for verification."""
    def __init__(self):
        self.calls = []

    def left_click_norm(self, x, y):
        self.calls.append(("left_click_norm", x, y))

    def double_click_norm(self, x, y):
        self.calls.append(("double_click_norm", x, y))

    def right_click_norm(self, x, y):
        self.calls.append(("right_click_norm", x, y))

    def type_text(self, text):
        self.calls.append(("type_text", text))

    def press_key(self, key):
        self.calls.append(("press_key", key))

    def hotkey(self, keys):
        self.calls.append(("hotkey", tuple(keys)))

    def scroll(self, amount):
        self.calls.append(("scroll", amount))

    def mouse_move_norm(self, x, y):
        self.calls.append(("mouse_move_norm", x, y))

    def mouse_down(self, button):
        self.calls.append(("mouse_down", button))

    def mouse_up(self, button):
        self.calls.append(("mouse_up", button))

    def drag_to_norm(self, x, y, button):
        self.calls.append(("drag_to_norm", x, y, button))


# Patch time.sleep to avoid actual delays
original_sleep = time.sleep
time.sleep = lambda secs: None

from src.actions import execute_action

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
# 1. VISIT_URL compound action
# ═══════════════════════════════════════════
print("\n=== VISIT_URL ===")
sb = MockSandbox()
execute_action(sb, {"action": "VISIT_URL", "url": "https://huggingface.co/models"})
# Should: hotkey ctrl+l, hotkey ctrl+a, type_text url, press enter
check("call count", len(sb.calls), 4)
check("1st: ctrl+l", sb.calls[0], ("hotkey", ("ctrl", "l")))
check("2nd: ctrl+a", sb.calls[1], ("hotkey", ("ctrl", "a")))
check("3rd: type url", sb.calls[2], ("type_text", "https://huggingface.co/models"))
check("4th: enter", sb.calls[3], ("press_key", "enter"))


# ═══════════════════════════════════════════
# 2. WEB_SEARCH compound action
# ═══════════════════════════════════════════
print("\n=== WEB_SEARCH ===")
sb = MockSandbox()
execute_action(sb, {"action": "WEB_SEARCH", "query": "best 7B models"})
check("call count", len(sb.calls), 4)
check("1st: ctrl+l", sb.calls[0], ("hotkey", ("ctrl", "l")))
check("2nd: ctrl+a", sb.calls[1], ("hotkey", ("ctrl", "a")))
check("3rd: type query", sb.calls[2], ("type_text", "best 7B models"))
check("4th: enter", sb.calls[3], ("press_key", "enter"))


# ═══════════════════════════════════════════
# 3. TYPE with click coordinates (Fara feature)
# ═══════════════════════════════════════════
print("\n=== TYPE with click_x/click_y ===")
sb = MockSandbox()
execute_action(sb, {"action": "TYPE", "text": "hello", "click_x": 0.3, "click_y": 0.5})
check("has click first", sb.calls[0], ("left_click_norm", 0.3, 0.5))
check("then type", sb.calls[1], ("type_text", "hello"))


# ═══════════════════════════════════════════
# 4. TYPE with delete_existing
# ═══════════════════════════════════════════
print("\n=== TYPE with delete_existing ===")
sb = MockSandbox()
execute_action(sb, {"action": "TYPE", "text": "new text", "delete_existing": True})
check("ctrl+a to select all", sb.calls[0], ("hotkey", ("ctrl", "a")))
check("then type", sb.calls[1], ("type_text", "new text"))


# ═══════════════════════════════════════════
# 5. TYPE with press_enter
# ═══════════════════════════════════════════
print("\n=== TYPE with press_enter ===")
sb = MockSandbox()
execute_action(sb, {"action": "TYPE", "text": "query", "press_enter": True})
check("type text", sb.calls[0], ("type_text", "query"))
check("then enter", sb.calls[1], ("press_key", "enter"))


# ═══════════════════════════════════════════
# 6. TYPE with all options
# ═══════════════════════════════════════════
print("\n=== TYPE with all options ===")
sb = MockSandbox()
execute_action(sb, {
    "action": "TYPE", "text": "full test",
    "click_x": 0.2, "click_y": 0.8,
    "delete_existing": True, "press_enter": True
})
check("1st: click field", sb.calls[0], ("left_click_norm", 0.2, 0.8))
check("2nd: ctrl+a", sb.calls[1], ("hotkey", ("ctrl", "a")))
check("3rd: type", sb.calls[2], ("type_text", "full test"))
check("4th: enter", sb.calls[3], ("press_key", "enter"))


# ═══════════════════════════════════════════
# 7. TYPE without press_enter (explicit false)
# ═══════════════════════════════════════════
print("\n=== TYPE without press_enter ===")
sb = MockSandbox()
execute_action(sb, {"action": "TYPE", "text": "no enter", "press_enter": False})
check("just type, no enter", len(sb.calls), 1)
check("type text", sb.calls[0], ("type_text", "no enter"))


# ═══════════════════════════════════════════
# 8. Standard CLICK (unchanged)
# ═══════════════════════════════════════════
print("\n=== Standard CLICK ===")
sb = MockSandbox()
execute_action(sb, {"action": "CLICK", "x": 0.5, "y": 0.5})
check("click", sb.calls[0], ("left_click_norm", 0.5, 0.5))


# ═══════════════════════════════════════════
# 9. MOVE
# ═══════════════════════════════════════════
print("\n=== MOVE ===")
sb = MockSandbox()
execute_action(sb, {"action": "MOVE", "x": 0.3, "y": 0.7})
check("move", sb.calls[0], ("mouse_move_norm", 0.3, 0.7))


# ═══════════════════════════════════════════
# 10. BITTI (no-op)
# ═══════════════════════════════════════════
print("\n=== BITTI ===")
sb = MockSandbox()
execute_action(sb, {"action": "BITTI"})
check("bitti no calls", len(sb.calls), 0)


# ═══════════════════════════════════════════
# 11. NOOP
# ═══════════════════════════════════════════
print("\n=== NOOP ===")
sb = MockSandbox()
execute_action(sb, {"action": "NOOP"})
check("noop no calls", len(sb.calls), 0)


# ═══════════════════════════════════════════
# 12. HOTKEY
# ═══════════════════════════════════════════
print("\n=== HOTKEY ===")
sb = MockSandbox()
execute_action(sb, {"action": "HOTKEY", "keys": ["alt", "left"]})
check("hotkey", sb.calls[0], ("hotkey", ("alt", "left")))


# ═══════════════════════════════════════════
# 13. SCROLL
# ═══════════════════════════════════════════
print("\n=== SCROLL ===")
sb = MockSandbox()
execute_action(sb, {"action": "SCROLL", "scroll": -3})
check("scroll", sb.calls[0], ("scroll", -3))


# ═══════════════════════════════════════════
# 14. Unknown action raises
# ═══════════════════════════════════════════
print("\n=== Unknown action ===")
sb = MockSandbox()
try:
    execute_action(sb, {"action": "UNKNOWN_ACTION_XYZ"})
    check("should have raised", False, True)
except ValueError as e:
    check("raises ValueError", "Unknown action" in str(e), True)


# Restore time.sleep
time.sleep = original_sleep

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
