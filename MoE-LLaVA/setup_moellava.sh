#!/usr/bin/env bash
uv sync
uv pip install --upgrade pip

uv pip install -e .
uv pip install -e ".[train]"

uv pip install ../DeepSpeed-0.9.5/
