# Phase 3: screen_changed False Negative Fixes

**Status**: Planned
**Priority**: MEDIUM — quick wins, fixes false feedback to models
**Effort**: Low

## Problem

The `screen_changed()` function in `src/vision.py` compares screenshots at 160x90
resolution using mean absolute pixel difference against `CHANGE_THRESHOLD=0.01`.

This produces false negatives:
- TYPE in address bar: diff=0.0002 (tiny text change) → marked "no effect"
- Applications menu open: diff=0.0049 → marked "no effect" (but it DID open)
- These false negatives give models wrong feedback ("your action failed" when it actually worked)

## Implementation Tasks

### 1. Skip screen_changed check for keyboard actions
TYPE, PRESS, and HOTKEY actions always "work" at the input level — the keystrokes
are sent. The visual change may be tiny (cursor blink, small text) or delayed.

In the agent loop, set `screen_changed=None` (unknown) for keyboard actions instead
of comparing screenshots. Only check for CLICK actions where "no effect" is meaningful.

### 2. Lower threshold for non-click actions (alternative)
If we want to keep checking: use `CHANGE_THRESHOLD * 0.3` for keyboard actions.

### 3. Add delay before comparison for animated UIs
Some UI changes are animated (dropdown opening, page loading). A small additional
delay (0.5s) before the comparison screenshot would catch these.

## Files to Modify
- `main.py` — agent loop screen_changed logic
- `gui_mission_control.py` — GUI agent loop (same change)
- `src/config.py` — optional: separate threshold for keyboard actions
