"""Agent core package."""
# agent/__init__.py
# 在 huggingface_hub 被 import 之前设置镜像，否则国内直连 huggingface.co 会卡死。
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")