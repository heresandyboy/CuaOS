# Phase 1: Fara-7B Integration Plan

**Status**: Ready to implement
**Priority**: HIGHEST — biggest impact, lowest risk
**Expected Impact**: 73.5% WebVoyager (vs ~40% UI-TARS), 2.5x fewer steps per task

## Why Fara-7B

- **Microsoft Research**, released Nov 2025, MIT license
- Built on **same Qwen2.5-VL-7B base** as UI-TARS → same `Qwen25VLChatHandler`
- 73.5% WebVoyager success rate (vs GPT-4o+SoM at 65.1%)
- ~16 steps/task average (vs ~41 for UI-TARS) — dramatically reduces loop risk
- Has native `visit_url` and `web_search` actions — no more clicking address bars!
- Q8_0 GGUF: ~8GB VRAM on RTX 4090

**Sources**:
- Model: https://huggingface.co/microsoft/Fara-7B
- GGUF: https://huggingface.co/bartowski/microsoft_Fara-7B-GGUF
- Code: https://github.com/microsoft/fara
- Paper: https://arxiv.org/abs/2511.19663
- Blog: https://www.microsoft.com/en-us/research/blog/fara-7b-an-efficient-agentic-model-for-computer-use/

## Technical Specification

### Output Format
Free-text reasoning + JSON tool call in XML tags:
```
[Chain-of-thought reasoning about what the agent sees]
<tool_call>
{"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [714, 448]}}
</tool_call>
```

### Parsing (from `fara_agent.py`)
```python
tmp = message.split("<tool_call>\n")
thoughts = tmp[0].strip()
action_text = tmp[1].split("\n</tool_call>")[0]
action = json.loads(action_text)
# action = {"name": "computer_use", "arguments": {"action": "...", ...}}
```

### Coordinate System
- **Absolute pixel coordinates** on the RESIZED image (same as UI-TARS!)
- Uses `smart_resize(height, width, factor=28)` — we already have this function
- System prompt tells model the resized dimensions dynamically
- Scale back: `x_orig = x_resized * (original_w / resized_w)`

### Action Vocabulary (11 actions)
| Fara Action | Args | CuaOS Internal Mapping |
|---|---|---|
| `left_click` | `coordinate: [x,y]` | `CLICK` with normalized x,y |
| `mouse_move` | `coordinate: [x,y]` | New: cursor move (execute as move) |
| `type` | `text, coordinate` | `TYPE` (optional: `press_enter`, `delete_existing_text`) |
| `key` | `keys: [str]` | `PRESS` (single key) or `HOTKEY` (multiple) |
| `scroll` | `pixels: int` | `SCROLL` (positive=up, negative=down) |
| `visit_url` | `url: str` | Compound: `HOTKEY ctrl+l` → `TYPE url` → `PRESS enter` |
| `web_search` | `query: str` | Compound: `HOTKEY ctrl+l` → `TYPE query` → `PRESS enter` |
| `history_back` | (none) | `HOTKEY alt+left` |
| `wait` | `time: int` | `WAIT` |
| `terminate` | `status: str` | `BITTI` |
| `pause_and_memorize_fact` | `fact: str` | Log only, no action |

### System Prompt Structure
ChatML format with Qwen function-call `<tools>` convention:
```
You are a web automation agent that performs actions on websites...
[Critical Points guidance]

<tools>
{"name": "computer_use", "description": "...\n* The screen's resolution is {W}x{H}.\n...",
 "parameters": {"properties": {"action": {"enum": [...]}, ...}}}
</tools>

For each function call, return a JSON object... within <tool_call></tool_call> XML tags
```

**Key**: Resolution in system prompt is DYNAMIC — computed per screenshot via `smart_resize()`.

### History Management
- All reasoning text + action tool calls retained across turns
- Only last N=3 screenshots kept (strip images from older messages)
- Multi-turn: alternating user (screenshot + observation) / assistant (thought + action)

### llama-cpp-python Setup
- Uses `Qwen25VLChatHandler` — **same handler we already use for UI-TARS**
- Temperature: 0.0 for best results
- n_ctx: 8192-16384 practical for GGUF
- Needs mmproj file from bartowski's GGUF repo

## Implementation Tasks

### 1. Add model profile to `src/config.py`
```python
"fara-7b": {
    "repo_id": "bartowski/microsoft_Fara-7B-GGUF",
    "model_file": "microsoft_Fara-7B-Q8_0.gguf",       # verify exact filename
    "mmproj_file": "microsoft_Fara-7B-mmproj-f16.gguf", # verify exact filename
    "chat_handler": "qwen25vl",  # same as UI-TARS
    "n_ctx": 16384,
    "n_batch": 512,
},
```
**Action**: Check bartowski's repo for exact filenames of Q8_0 and mmproj files.

### 2. Create `_build_fara_system_prompt()` in `src/llm_client.py`
- Construct the system prompt with `<tools>` block
- Inject dynamic resolution from `smart_resize()`
- Include Fara's detailed interaction tips (click center, handle popups, etc.)
- Include Critical Points guidance

### 3. Create `_parse_fara_output()` in `src/llm_client.py`
- Split on `<tool_call>\n` to get thoughts and action JSON
- Parse JSON from between `<tool_call>` and `</tool_call>` tags
- Convert Fara actions to CuaOS internal format:
  - `left_click` → normalize coords (divide by smart_resize dims) → `CLICK`
  - `visit_url` → execute as compound action sequence
  - `web_search` → execute as compound action sequence
  - `key` → map key names (Enter→enter, Control→ctrl, etc.)
  - `terminate` → `BITTI`
- Handle `pause_and_memorize_fact` as log-only

### 4. Create `_ask_fara()` in `src/llm_client.py`
- Build multi-turn message history (not single-shot like current models)
- Keep last 3 screenshots in conversation
- Include observation text from previous actions
- Use temperature=0.0, max_tokens=1024

### 5. Create `_format_fara_history()` for multi-turn messages
- Convert CuaOS history into Fara's expected alternating user/assistant format
- User messages: screenshot + observation text
- Assistant messages: thoughts + `<tool_call>` action
- Strip images from messages older than N=3

### 6. Update `ask_next_action()` dispatcher
- Add `"fara"` chat_handler type routing
- Route to `_ask_fara()` when `cfg.CHAT_HANDLER == "fara"` or similar

### 7. Handle compound actions (`visit_url`, `web_search`)
- These map to multiple sequential CuaOS actions
- Option A: Execute the compound sequence internally in actions.py
- Option B: Convert to the first action (HOTKEY ctrl+l) and let the model continue
- **Recommendation**: Option A — execute full sequence, it's what makes Fara efficient

### 8. Update GUI model selector
- Add Fara-7B to the model dropdown in `gui_mission_control.py`

### 9. Update `src/config.py` allowed keys/hotkeys
- Fara may emit key names we don't currently allow (ArrowUp vs up, etc.)
- Add mapping: `{"Enter": "enter", "ArrowUp": "up", "ArrowDown": "down", ...}`

### 10. Test and compare
- Run same task with Fara-7B Q8_0 vs UI-TARS Q8_0
- Compare: steps to completion, loop frequency, success rate

## Key Differences from Current UI-TARS Integration

| Aspect | UI-TARS | Fara-7B |
|---|---|---|
| Output format | `Thought: ... Action: click(...)` | `[reasoning] <tool_call>JSON</tool_call>` |
| Coordinate system | Pixel on smart_resize dims | Same (pixel on smart_resize dims) |
| History | Single-shot (all history in instruction) | Multi-turn conversation (alternating messages) |
| URL navigation | Must click address bar + type | Native `visit_url` action |
| Search | Must click address bar + type | Native `web_search` action |
| Termination | `finished(content='...')` | `terminate(status='success')` |
| Temperature | 0.0 | 0.0 |
| System prompt | Custom instruction text | ChatML with `<tools>` function schema |

## Risk Assessment
- **Low risk**: Same Qwen2.5-VL base, same chat handler, same smart_resize
- **Medium effort**: New output parser, multi-turn history, compound actions
- **High confidence**: Well-documented format, MIT license, active development
