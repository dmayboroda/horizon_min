---
title: Duck Hunt VLM Demo
emoji: "\U0001F3AF"
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "5.29.0"
app_file: app.py
hardware: t4-small
python_version: "3.12"
models:
  - dmayboroda/dh_ministal_gpro
  - mistralai/Ministral-3-8B-Instruct-2512-BF16
pinned: false
---

# Duck Hunt VLM Demo

Watch a vision-language model play Duck Hunt! The model (Ministral-3B + LoRA, trained with GRPO) sees game frames, predicts duck trajectories accounting for processing latency, and fires.

- **Model**: [dmayboroda/dh_ministal_gpro](https://huggingface.co/dmayboroda/dh_ministal_gpro)
- **Hit rate**: 60.9% after training
- **Input**: 4 sequential 512x512 game frames
- **Output**: Normalized (x, y) shot coordinates + horizon prediction
