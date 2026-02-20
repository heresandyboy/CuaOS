# Phase 2: API Planner Recovery System

**Status**: Planned
**Priority**: HIGH — breaks the nudge→same-mistake cycle
**Prerequisite**: Phase 1 (Fara-7B) should be tested first
**Existing code**: `src/planner.py` already has OpenRouter + llama_index setup

## Architecture

When the guard system detects a loop (nudge triggered), instead of just injecting
text feedback that the local model ignores, call an API model via OpenRouter to
analyze the situation and provide a specific recovery action.

```
Guard detects loop
  → Capture: current screenshot + action history + what's been tried
  → Call API planner (Claude/GPT-4o via OpenRouter)
  → Planner analyzes and returns specific action (e.g., "visit_url('https://huggingface.co/models?sort=downloads')")
  → Agent executes planner's action directly (bypasses local model for 1 step)
  → Resume local model after planner action
```

## Implementation Tasks

### 1. Add `generate_recovery_plan()` to `src/planner.py`
- Input: screenshot (base64), action history, guard message, objective
- Output: specific action dict that can be executed immediately
- Use a vision-capable API model (Claude 3.5 Sonnet, GPT-4o)

### 2. Add recovery prompt template
- Show the API model the screenshot + history + what failed
- Ask for ONE specific action to break the loop
- Format response as CuaOS action dict

### 3. Integrate with guard system in agent loop
- When `check_repeat()` returns `NUDGE`:
  - If API planner configured: call planner instead of text feedback
  - If no API key: fall back to current text feedback behavior
- Execute planner's action directly, skip local model for that step

### 4. Configuration
- `PLANNER_RECOVERY_ENABLED: bool = False` (opt-in)
- `PLANNER_MAX_CALLS_PER_RUN: int = 3` (cost control)
- Use existing `PLANNER_PROVIDER`, `PLANNER_API_KEY`, `PLANNER_MODEL`

### 5. GUI integration
- Show "Planner Recovery" in the GUI when API model is called
- Display planner's reasoning

## Sources
- Agent S2/S3 uses exactly this pattern: https://github.com/simular-ai/Agent-S
- Existing planner code: `src/planner.py`
- Config: `PLANNER_PROVIDER`, `PLANNER_API_KEY`, `PLANNER_MODEL` in `src/config.py`
