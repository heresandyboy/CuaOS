# Phase 4: OmniParser V2 Integration

**Status**: Future (only if Fara-7B alone isn't sufficient)
**Priority**: LOW — significant effort, may not be needed
**Effort**: High

## What It Does

OmniParser V2 (Microsoft) detects all interactive UI elements in a screenshot using
YOLOv8 + Florence-2, labels them with IDs, and feeds structured data to any LLM.

This eliminates the "clicking slightly wrong coordinates" problem by converting
coordinate prediction into element selection by ID.

## Architecture
1. Screenshot → YOLOv8 (~100MB) detects elements
2. Florence-2 (~200MB) captions each element
3. Set-of-Mark overlay: numbered bounding boxes on screenshot
4. Output: annotated image + JSON element list
5. LLM receives structured data, selects by element ID
6. Action coordinates = center of selected element's bounding box

## Performance
- Latency: 0.8s/frame on RTX 4090
- VRAM: ~10GB
- Combined with Fara-7B Q8_0 (~8GB): ~18GB total, fits on RTX 4090

## Dependencies
- ultralytics (YOLOv8)
- Florence-2 model (from HuggingFace)
- ~10GB additional VRAM

## Sources
- GitHub: https://github.com/microsoft/OmniParser
- HuggingFace: microsoft/OmniParser-v2.0
- Blog: https://www.microsoft.com/en-us/research/articles/omniparser-v2-turning-any-llm-into-a-computer-use-agent/

## Note
Fara-7B was designed to predict coordinates natively WITHOUT a parser. Microsoft's
own benchmarks show Fara-7B alone outperforming GPT-4o+SoM on WebVoyager (73.5% vs
65.1%). So this phase may be unnecessary.
