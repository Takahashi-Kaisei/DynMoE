# MoE-LLaVA StableLM Colab Handoff

Date: 2026-05-08

## Goal

Run a sanity check for the StableLM-based MoE-LLaVA model on Google Colab, including
automatic model download and one successful image-grounded inference pass.

Target repo:

- `/content/DynMoE`

Target model:

- `LanguageBind/MoE-LLaVA-StableLM-1.6B-4e`

Current Colab status:

- Completed successfully on a Tesla T4 runtime.
- The model downloaded automatically.
- DeepSpeed reached the interactive `USER:` prompt.
- One inference pass succeeded with the repo sample image.

## Why Colab Is Needed

The local machine used so far is an Apple Silicon MacBook Air (`Mac14,2`, Apple M2,
16 GB RAM) without NVIDIA CUDA. `MoE-LLaVA`'s StableLM MoE path calls
`deepspeed.init_distributed(dist_backend='nccl')` in
`MoE-LLaVA/moellava/model/builder.py`, so the intended inference path is
CUDA/NCCL-oriented.

Local validation already completed:

- Python 3.10 uv environment creation
- installation of `torch==2.0.1`, `transformers==4.37.0`, `moellava`, and local
  `DeepSpeed-0.9.5`
- import checks for `torch`, `deepspeed`, `moellava`, and `moellava.model.builder`
- StableLM tokenizer load with `Arcade100kTokenizer`

Local blocker:

- non-CUDA environment
- DeepSpeed distributed path expects NCCL / MPI-style runtime

## Recommended Colab Environment

Use a GPU runtime in Colab.

Suggested checks:

```bash
!nvidia-smi
!python --version
```

The successful run exposed:

```text
GPU: Tesla T4
Driver Version: 580.82.07
CUDA Version: 13.0
Memory: 15360 MiB
```

Prefer Python 3.10 for this pinned dependency stack. In the successful run, Colab's
default `python` was `3.12.13`, while `/usr/bin/python3.10` was available.
`torch==2.0.1` should not be installed into the Python 3.12 environment.

If Python 3.10 exists but has no pip/venv support, install the OS packages first:

```bash
!apt-get update
!apt-get install -y python3.10-venv python3.10-dev
```

The successful run used a dedicated venv:

```bash
%cd /content/DynMoE
!python3.10 -m venv .venv310
```

## Setup Steps

Clone the repo if needed. If `/content/DynMoE` already exists, skip `git clone`.

```bash
%cd /content
!git clone https://github.com/LINs-lab/DynMoE.git
%cd /content/DynMoE
```

Install the tested stack into the Python 3.10 venv:

```bash
!python3.10 -m venv .venv310
!.venv310/bin/python -m pip --log /tmp/pip.log install --upgrade pip
!.venv310/bin/python -m pip --log /tmp/pip.log install torch==2.0.1 torchvision==0.15.2
!.venv310/bin/python -m pip --log /tmp/pip.log install transformers==4.37.0 tokenizers==0.15.1 sentencepiece==0.1.99
!.venv310/bin/python -m pip --log /tmp/pip.log install accelerate==0.21.0 peft==0.4.0 openpyxl==3.1.2
!.venv310/bin/python -m pip --log /tmp/pip.log install -e ./MoE-LLaVA
!.venv310/bin/python -m pip --log /tmp/pip.log install -e ./DeepSpeed-0.9.5
```

`--log /tmp/pip.log` avoids noisy failures from pip trying to write
`/var/log/pip.log` in restricted environments.

If `flash-attn` is needed later, install it only after the base sanity check. It was
not required for this check.

## Sanity Check Command

Use the repo sample image:

```bash
%cd /content/DynMoE/MoE-LLaVA
!HF_HOME=/content/hf_cache TRANSFORMERS_CACHE=/content/hf_cache \
  /content/DynMoE/.venv310/bin/deepspeed --include localhost:0 moellava/serve/cli.py \
  --model-path "LanguageBind/MoE-LLaVA-StableLM-1.6B-4e" \
  --image-file "assets/image.jpg"
```

`HF_HOME=/content/hf_cache` avoids Hugging Face cache permission warnings under
`/root/.cache/huggingface/hub`. `TRANSFORMERS_CACHE` emits a deprecation warning in
Transformers 4.37, but it worked; `HF_HOME` is the better long-term setting.

After the CLI prompt appears, enter:

```text
Describe this image.
```

Success condition:

- model downloads automatically
- tokenizer and processor initialize
- model loads without crash
- one text response is generated

Observed successful response:

```text
The image features a large red heart painted on a blue door, with the heart taking up most of the door's surface. The heart is a prominent and eye-catching element in the scene. The door appears to be a part of a building or a structure, and it is located near a blue wall.
```

The CLI does not treat `exit` as a command after inference; it may generate text for
`exit`. Use Colab's stop button or `Ctrl-C` to terminate it.

## Verification Commands

From `/content/DynMoE`:

```bash
!.venv310/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
!.venv310/bin/python -c "import deepspeed; print(deepspeed.__version__)"
!.venv310/bin/python -c "import moellava; print('moellava ok')"
!.venv310/bin/python -c "import moellava.model.builder; print('builder ok')"
!nvidia-smi
```

Successful versions/checks from the run:

```text
torch: 2.0.1+cu117
torch.cuda.is_available(): True
deepspeed: 0.9.5+4927232
moellava ok
builder ok
```

If `torch.cuda.is_available()` is false, check whether the runtime is actually using
GPU and whether the command runner has permission to access the NVIDIA driver.

## Expected Warnings

These warnings appeared but did not block successful inference:

- `TRANSFORMERS_CACHE` deprecation warning; prefer `HF_HOME` going forward.
- `NCCL backend in DeepSpeed not yet implemented`, followed by TorchBackend NCCL
  initialization.
- `TypedStorage is deprecated`.
- generation warning about missing `attention_mask` and `pad_token_id`.
- `bitsandbytes` CPU/GPU support warnings during import.

## Troubleshooting

Check these in order:

1. `deepspeed` import and version
2. CUDA visibility via `nvidia-smi`
3. `torch.cuda.is_available()`
4. Hugging Face download/auth/network errors
5. whether the command is running interactively; non-interactive stdin may close
   before the prompt can be answered
6. GPU OOM during model load

If OOM occurs, try lower-memory loading options before changing the model family.

## Practical Tips

- Keep using `/content/DynMoE/.venv310`; do not use the default Python 3.12
  environment.
- Run model commands from `/content/DynMoE/MoE-LLaVA`, because the sample image path is
  `assets/image.jpg`.
- Use `HF_HOME=/content/hf_cache` to avoid cache permission issues.
- Use TTY/interactive execution for `moellava/serve/cli.py`; otherwise stdin may close
  before a prompt can be answered.
- If the model is already cached, shard download should be instant.
- T4 15 GB was enough for this sanity check.
- After inference, terminate the CLI with `Ctrl-C`.
