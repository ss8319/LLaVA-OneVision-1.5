# LLaVA-OneVision-1.5 - Setup & Usage Guide

## Quick Setup

```bash
cd /mnt/hdd/sda/samus/LLaVA-OneVision-1.5
source $HOME/.local/bin/env  # if uv isn't in your PATH
source .venv/bin/activate
```

## Environment Setup

### 1. Install uv (Fast Python Package Manager)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env  # Add to PATH
```

### 2. Create Virtual Environment & Install Dependencies
```bash
cd /mnt/hdd/sda/samus/LLaVA-OneVision-1.5
source $HOME/.local/bin/env
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 3. Install DeepSpeed (for training)
```bash
source .venv/bin/activate
uv pip install deepspeed
```

**Note**: DeepSpeed may show CUDA warnings on import - this is normal if CUDA_HOME isn't set. Training will still work.

## Key Files & Directories

### Training Infrastructure
- **DeepSpeed (Simple)**: `ds/` - For easy fine-tuning with 4-8 GPUs
  - Training: `ds/src/train/train_sft.py`
  - Scripts: `ds/scripts/finetune.sh`, `ds/scripts/pretrain.sh`
  - Inference: `ds/inference.py`
  
- **Megatron (Production)**: `aiak_megatron/` + `aiak_training_llm/` - For large-scale training

### CUDA Error: Device-side assert triggered
**Problem**: `Assertion 'probability tensor contains either 'inf', 'nan' or element < 0' failed`

**Root Cause**: Model uses sampling by default which causes numerical instability

**Solution**: Use greedy decoding (`do_sample=False`) in generation calls.

