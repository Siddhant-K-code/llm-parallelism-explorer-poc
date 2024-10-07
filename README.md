# LLM Parallelism Explorer PoC

LLM Parallelism Explorer PoC is a cutting-edge research tool designed to optimize parallelism strategies for large language models, with a particular focus on Mixture of Experts (MoE) architectures. This proof-of-concept project performs a comprehensive search across various parallelism configurations to estimate memory usage and identify optimal setups for efficient training on distributed systems. It also uses [facebookresearch/hydra](https://github.com/facebookresearch/hydra) for easy configuration management.

## Features

- Supports advanced parallelism techniques:
  - Tensor Parallelism (TP)
  - Pipeline Parallelism (PP)
  - Expert Parallelism (EP)
  - Context Parallelism (CP)
  - Data Parallelism (DP)
- Precise memory estimation for model parameters, optimizer states, and activations
- Flexible search space for parallelism configurations
- Multiple data parallel sharding strategies
- CSV output for in-depth analysis of results
- Hydra-powered configuration management

## Installation

Install the required dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

Use a specific configuration file and customize the GPU range:

```bash
python main.py \
    --config-name llama3.1-405b.yaml \
    +ngpus_range="[8, 128, 1024, 10240]"
```

## Configuration

Leverage Hydra for easy configuration management. Modify these parameters in your YAML config file:

- Model architecture details (e.g., hidden size, number of layers)
- MoE-specific settings (e.g., number of experts, expert frequency)
- Training parameters (e.g., global batch size, data types)
- Parallelism search ranges (e.g., TP, PP, EP ranges)

## Output

The script generates memory_estimation.csv with comprehensive memory estimations for each valid parallelism configuration, including:

- Total memory usage
- Model and optimizer states memory
- Activations memory
- Expert and non-expert parameters
- Component-specific activation memory

## Visualization

This visualization of memory estimation results helps identify optimal configurations at a glance.

## Credits

This project builds upon state-of-the-art parallelism techniques from recent research:

- [Tensor Parallelism: Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism" (2019)](https://arxiv.org/abs/1909.08053)
- [Pipeline Parallelism: Huang et al., "GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism" (2019)](https://arxiv.org/abs/1811.06965)
- [Expert Parallelism: Lepikhin et al., "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding" (2020)](https://arxiv.org/abs/2006.16668)
- [Context Parallelism: Korthikanti et al., "Reducing Activation Recomputation in Large Transformer Models" (2022)](https://arxiv.org/abs/2205.05198)
