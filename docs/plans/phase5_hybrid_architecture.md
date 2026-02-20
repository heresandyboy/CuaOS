# Phase 5: Full Hybrid Architecture

**Status**: Future (long-term goal)
**Priority**: LOW — major architectural redesign
**Effort**: Very High

## Goal
Approach Agent S2/S3 level performance (72.6% OSWorld, surpassing human-level)
by implementing a Manager/Worker hierarchy with specialized grounding.

## Architecture (based on Agent S2/S3)
- **Manager** (generalist planning): High-level task decomposition
- **Worker** (execution): Step-level action selection
- **Mixture of Grounding (MoG)**: Routes grounding to specialized experts
- **Proactive Hierarchical Planning**: Dynamic plan updates after each subtask

## Key Models to Consider
- **OpenCUA-7B**: Full framework, single GPU, covers 3 OSes, 200+ apps
  - GitHub: https://github.com/xlang-ai/OpenCUA
- **GUI-Actor-7B**: Attention-based action head on Qwen2.5-VL, 44.6% ScreenSpot-Pro
  - HuggingFace: microsoft/GUI-Actor-7B-Qwen2.5-VL
- **ShowUI-2B**: Lightweight, runs without GPU
  - GitHub: https://github.com/showlab/ShowUI

## Benchmark Targets
| Benchmark | Current | Target |
|---|---|---|
| WebVoyager | ~0% (fails) | >70% |
| ScreenSpot-v2 | unknown | >80% |
| Task step count | 41+ | <20 |

## Sources
- Agent S2/S3: https://github.com/simular-ai/Agent-S
- Agent S2 paper: https://arxiv.org/html/2504.00906v1
- OSWorld benchmark: https://os-world.github.io/
