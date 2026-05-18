#!/usr/bin/env bash
uv sync
uv pip install --upgrade pip

uv pip install -e .
uv pip install -e ".[train]"

uv pip install ../DeepSpeed-0.9.5/

uv pip install "setuptools<82"

uv pip install "flash-attn==2.3.5" --no-build-isolation

uv tool install huggingface_hub
