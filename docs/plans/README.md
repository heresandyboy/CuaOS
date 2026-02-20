# CuaOS Implementation Plans

## Roadmap Overview

| Phase | Plan | Priority | Status |
|---|---|---|---|
| 1 | [Fara-7B Integration](fara7b_integration.md) | HIGHEST | Ready to implement |
| 2 | [API Planner Recovery](phase2_api_planner_recovery.md) | HIGH | Planned |
| 3 | [screen_changed Fixes](phase3_screen_changed_fixes.md) | MEDIUM | Planned |
| 4 | [OmniParser V2](phase4_omniparser_v2.md) | LOW | Future |
| 5 | [Hybrid Architecture](phase5_hybrid_architecture.md) | LOW | Future |

## Research (stored in Serena memories)
- `research_gui_agent_benchmarks` — WebVoyager, OSWorld, ScreenSpot-Pro leaderboards
- `research_fara_7b` — Fara-7B overview and benchmarks
- `research_fara_7b_integration_spec` — Exact output format, coordinate system, parsing code
- `research_omniparser_v2` — OmniParser V2 architecture and integration
- `research_hybrid_architectures` — Agent S2/S3, GUI-Actor, hybrid patterns
- `research_quantization_guide` — GGUF quantization cliff for coordinate prediction
- `research_current_issues` — Diagnosed problems with Qwen3-VL and UI-TARS runs

## Key Decision
Phase 1 (Fara-7B) is the highest-impact, lowest-risk change. Same Qwen2.5-VL base
and chat handler as UI-TARS, but 73.5% WebVoyager success rate and 2.5x fewer steps.
Has native `visit_url` and `web_search` actions that eliminate the most common failure
mode (trying to click address bars and search results).
