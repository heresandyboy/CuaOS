# planner.py — Planning LLM via OpenRouter (llama_index)
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from llama_index.llms.openrouter import OpenRouter
from llama_index.core.llms import ChatMessage


# ─── Planner Configuration ───────────────────────────────────────────

@dataclass
class PlannerConfig:
    provider: str = "openrouter"           # openrouter | openai | local
    api_key: str = ""
    model: str = ""
    max_tokens: int = 1024
    api_url: str = "https://openrouter.ai/api/v1/chat/completions"


# ─── System Prompt ────────────────────────────────────────────────────

PLANNER_SYSTEM_PROMPT = """\
You are a Computer Use Agent task planner. The user gives you a simple, high-level \
command about what they want to do on a Linux desktop (XFCE). Your job is to break \
it down into a detailed, step-by-step action plan that another AI agent will execute.

RULES:
1. Output ONLY comma-separated action steps. No extra explanation.
2. Each step MUST use one of these verbs:
   - click [target]
   - double_click [target]
   - right_click [target]
   - type [text]
   - press [key]
   - hotkey [key1+key2]
   - scroll [up/down]
   - wait [1 Step]
3. [target] should describe a visible UI element (e.g. "browser icon on taskbar", "address bar", "search button").
4. [text] is the literal text to type.
5. [key] is a key name (enter, tab, esc, backspace, etc.).
6. Keep the plan concise: only essential steps to accomplish the task.
7. Add "wait" after actions that trigger loading (opening apps, navigating to URLs).
8. Do NOT add numbering, bullets, or newlines between steps. Use commas only.

EXAMPLE INPUT: "Open YouTube"
EXAMPLE OUTPUT: click browser icon on taskbar, wait, click address bar, type youtube.com, press enter, wait

EXAMPLE INPUT: "Open terminal and create a folder called projects"
EXAMPLE OUTPUT: click terminal icon on taskbar, wait, type mkdir projects, press enter

EXAMPLE INPUT: "Search Wikipedia for artificial intelligence"
EXAMPLE OUTPUT: click browser icon on taskbar, wait, click address bar, type wikipedia.org, press enter, wait, click search input field, type artificial intelligence, press enter, wait
"""


# ─── Create Planner LLM Instance ─────────────────────────────────────

def create_planner(config: PlannerConfig) -> Optional[OpenRouter]:
    """Create an OpenRouter LLM instance for planning."""
    if not config.api_key:
        return None
    if config.provider == "local":
        return None

    return OpenRouter(
        api_key=config.api_key,
        max_tokens=config.max_tokens,
        context_window=4096,
        model=config.model,
    )


# ─── Generate Plan ───────────────────────────────────────────────────

def generate_plan(planner: OpenRouter, objective: str) -> List[str]:
    """
    Send the user's simple command to the planning LLM.
    Returns a list of action step strings.
    """
    messages = [
        ChatMessage(role="system", content=PLANNER_SYSTEM_PROMPT),
        ChatMessage(role="user", content=objective),
    ]

    resp = planner.chat(messages)
    raw_plan = resp.message.content.strip()

    # Parse comma-separated steps
    steps = [s.strip() for s in raw_plan.split(",") if s.strip()]
    return steps


def parse_plan_step(step: str) -> dict:
    """
    Parse a single plan step string into a structured dict.
    Used for display/logging purposes.

    Examples:
        "click browser icon" -> {"verb": "click", "target": "browser icon"}
        "type youtube.com"   -> {"verb": "type", "target": "youtube.com"}
        "press enter"        -> {"verb": "press", "target": "enter"}
        "wait"               -> {"verb": "wait", "target": ""}
    """
    step = step.strip().lower()

    verbs = ["double_click", "right_click", "click", "type", "press", "hotkey", "scroll", "wait"]

    for verb in verbs:
        if step.startswith(verb):
            target = step[len(verb):].strip()
            return {"verb": verb, "target": target}

    # Fallback: treat entire step as a custom instruction
    return {"verb": "custom", "target": step}
