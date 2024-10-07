import csv
from collections import namedtuple
from itertools import product
from typing import Dict, List

import hydra
from omegaconf import DictConfig, open_dict

ParallelConfig = namedtuple("ParallelConfig", ["ngpus", "tp", "pp", "ep", "cp", "dp"])
MemoryEstimation = namedtuple(
    "MemoryEstimation",
    [
        "name",
        "ngpus",
        "TP",
        "PP",
        "EP",
        "CP",
        "DP",
        "sub_seq_length",
        "micro_batch_size",
        "num_of_micro_batches",
        "pipeline_parallelism_bubble_rate",
        "data_parallel_sharding_strategy",
        "total_memory_gb",
        "model_and_optimizer_states_memory_gb",
        "activations_memory_gb",
        "cross_entropy_activation_memory_gb",
        "practice_activations_memory_gb",
        "expert_parameters_m",
        "non_expert_parameters_m",
        "expert_layers_activations_gb",
        "practice_expert_layers_activations_gb",
        "non_expert_layers_activations_gb",
        "mlp_activation_mb",
        "moe_activation_mb",
        "token_imbalance_hypothesis",
    ],
)


def calculate_model_state_coefficient(
    trainer_config: DictConfig, data_parallel_size: int
) -> float:
    """Calculate the coefficient of the model state size with respect to the parameter quantity."""
    shard_strategy = trainer_config.data_parallel_sharding_strategy
    param_dtype = trainer_config.param_dtype
    grad_dtype = trainer_config.grad_dtype
    d = data_parallel_size
    psi_table = {
        "NO_OP": {
            ("float16", "float16"): 20.0,
            ("float16", "float32"): 18.0,
            ("float32", "float32"): 16.0,
        },
        "OPTIMIZER_STATES": {
            ("float16", "float16"): 4 + 16.0 / d,
            ("float16", "float32"): 6 + 12.0 / d,
            ("float32", "float32"): 8 + 8.0 / d,
        },
        "OPTIMIZER_STATES_AND_GRADS": {
            ("float16", "float16"): 2 + 18.0 / d,
            ("float16", "float32"): 2 + 16.0 / d,
            ("float32", "float32"): 4 + 12.0 / d,
        },
        "FULLY_SHARD": {
            ("float16", "float16"): 20.0 / d,
            ("float16", "float32"): 16.0 / d,
            ("float32", "float32"): 16.0 / d,
        },
    }
    return psi_table[shard_strategy][(param_dtype, grad_dtype)]


def estimate_memory(
    model_config: DictConfig,
    trainer_config: DictConfig,
    parallel_config: ParallelConfig,
    token_imbalance_hypothesis: float = 1.0,
) -> MemoryEstimation:
    """Estimate memory usage for a given model and parallel configuration."""
    ngpus, tp, pp, ep, cp, dp = parallel_config
    assert (
        tp * pp * cp * dp == ngpus
    ), "Parallelism configuration does not match the number of GPUs."
    assert dp % ep == 0, "Data parallelism must be divisible by expert parallelism."
    data_module_expert_parallelism = dp // ep
    share_embeddings_and_output_weights = model_config.get(
        "share_embeddings_and_output_weights", True
    )

    v, h, h_ffn, nlayers = (
        model_config.vocab_size,
        model_config.hidden_size,
        model_config.ffn_hidden_size,
        model_config.num_layers,
    )
    assert (
        nlayers % pp == 0
    ), "Number of layers must be divisible by pipeline parallelism."
    assert (
        model_config.seq_length % cp == 0
    ), "Sequence length must be divisible by context parallelism world size."
    s, b, m = (
        model_config.seq_length // cp,
        model_config.micro_batch_size,
        model_config.global_batch_size / dp / model_config.micro_batch_size,
    )
    activation = model_config.activation

    assert trainer_config.param_dtype == "float16", "Only float16 is supported for now."
    assert trainer_config.grad_dtype == "float32", "Only float32 is supported for now."

    f_expert, k, n_experts = (
        model_config.moe.expert_frequency,
        model_config.moe.k,
        model_config.moe.num_experts,
    )
    assert (
        n_experts % ep == 0
    ), "Number of experts must be divisible by number of expert parallelism."
    c = token_imbalance_hypothesis if ep != 1 else 1.0  # expert capacity factor

    # Calculate model parameters
    embedding_parameters = (
        v * h // tp * (2 if pp == 1 and not share_embeddings_and_output_weights else 1)
    )
    if hasattr(model_config, "num_key_value_heads"):
        d_head = h // model_config.num_attention_heads
        n_kv_heads, n_q_heads = (
            model_config.num_key_value_heads,
            model_config.num_attention_heads,
        )
        attention_layer_parameters = (
            h // tp * (n_q_heads * d_head + 2 * n_kv_heads * d_head + h)
        )
    else:
        attention_layer_parameters = h * h * 4 // tp
    mlp_layer_parameters = h * h_ffn * (3 if activation == "swiglu" else 2) // tp
    non_expert_parameters = (
        embedding_parameters
        + (attention_layer_parameters + mlp_layer_parameters * (1.0 - f_expert))
        * nlayers
        // pp
    )
    expert_parameters = (
        (mlp_layer_parameters * f_expert * n_experts // ep) * nlayers // pp
    )

    # Calculate memory usage
    non_expert_psi = calculate_model_state_coefficient(trainer_config, dp)
    expert_psi = calculate_model_state_coefficient(
        trainer_config, data_module_expert_parallelism
    )
    model_and_optimizer_states_memory = (
        non_expert_parameters * non_expert_psi + expert_parameters * expert_psi
    )

    cross_entropy_activation = (2 + 4 + 4) * s * b * v // tp
    mlp_activation = 2 * s * b * (h + (3 if activation == "swiglu" else 2) * h_ffn) / tp
    moe_activation = 2 * (s * b * h + c * k * s * b * h) // tp + c * k * mlp_activation
    practice_moe_activation = (
        s * b * h // tp + 4 * c * k * s * b * h + c * k * mlp_activation
    )
    expert_layers_activations = nlayers * f_expert * moe_activation
    practice_expert_layers_activations = nlayers * f_expert * practice_moe_activation
    non_expert_layers_activations = nlayers * (
        (1.0 - f_expert) * mlp_activation + 2 * 7 * s * b * h / tp
    ) + (cross_entropy_activation if pp == 1 else 0.0)
    activations_memory = expert_layers_activations + non_expert_layers_activations
    practice_activations_memory = (
        practice_expert_layers_activations + non_expert_layers_activations
    )

    pipeline_parallelism_bubble_rate = (pp - 1) / m

    return MemoryEstimation(
        name=model_config.name,
        ngpus=ngpus,
        TP=tp,
        PP=pp,
        EP=ep,
        CP=cp,
        DP=dp,
        sub_seq_length=model_config.seq_length // cp,
        micro_batch_size=model_config.micro_batch_size,
        num_of_micro_batches=m,
        pipeline_parallelism_bubble_rate=pipeline_parallelism_bubble_rate,
        data_parallel_sharding_strategy=trainer_config.data_parallel_sharding_strategy,
        total_memory_gb=(model_and_optimizer_states_memory + activations_memory)
        / 1024**3,
        model_and_optimizer_states_memory_gb=model_and_optimizer_states_memory
        / 1024**3,
        activations_memory_gb=activations_memory / 1024**3,
        cross_entropy_activation_memory_gb=cross_entropy_activation / 1024**3,
        practice_activations_memory_gb=practice_activations_memory / 1024**3,
        expert_parameters_m=expert_parameters / 1024**2,
        non_expert_parameters_m=non_expert_parameters / 1024**2,
        expert_layers_activations_gb=expert_layers_activations / 1024**3,
        practice_expert_layers_activations_gb=practice_expert_layers_activations
        / 1024**3,
        non_expert_layers_activations_gb=non_expert_layers_activations / 1024**3,
        mlp_activation_mb=mlp_activation / 1024**2,
        moe_activation_mb=moe_activation / 1024**2,
        token_imbalance_hypothesis=token_imbalance_hypothesis,
    )


def get_model_parallel_search_space(cfg: DictConfig) -> Dict[str, List]:
    """Get the model parallel search space from the configuration."""
    num_experts = cfg.model.moe.num_experts
    search_space = {
        "mbs_range": [1, 2, 4],
        "cp_range": [1, 2, 4, 8],
        "ep_range": [2**i for i in range(int(num_experts).bit_length())]
        if num_experts > 0
        else [1],
        "tp_range": [1, 2, 4, 8],
        "pp_range": [1, 2, 4, 8, 16],
        "ngpus_range": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 10240],
        "shard_strategys": [
            "NO_OP",
            "OPTIMIZER_STATES",
            "OPTIMIZER_STATES_AND_GRADS",
            "FULLY_SHARD",
        ],
    }

    for key in search_space:
        if key in cfg:
            search_space[key] = cfg[key]

    return search_space


@hydra.main(version_base="1.1", config_path="./", config_name="Mixtral_8x7b.yaml")
def main(cfg: DictConfig):
    with open("memory_estimation.csv", "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(MemoryEstimation._fields)

        if not hasattr(cfg.model, "moe"):
            with open_dict(cfg.model):
                cfg.model.moe = DictConfig(
                    {
                        "expert_frequency": 0,
                        "k": 0,
                        "num_experts": 0,
                        "token_imbalance_hypothesis": 1.0,
                    }
                )

        mp_space = get_model_parallel_search_space(cfg)
        for ngpus, tp, pp, ep, cp, dp_sharding_strategy, mbs in product(
            mp_space["ngpus_range"],
            mp_space["tp_range"],
            mp_space["pp_range"],
            mp_space["ep_range"],
            mp_space["cp_range"],
            mp_space["shard_strategys"],
            mp_space["mbs_range"],
        ):
            mp = tp * pp * cp
            if ngpus % mp != 0 or cfg.model.num_layers % pp != 0:
                continue
            dp = ngpus // mp
            if dp % ep != 0 or cfg.model.global_batch_size % (dp * mbs) != 0:
                continue

            cfg.model.micro_batch_size = mbs
            cfg.trainer.data_parallel_sharding_strategy = dp_sharding_strategy

            parallel_config = ParallelConfig(
                ngpus=ngpus, tp=tp, pp=pp, ep=ep, cp=cp, dp=dp
            )
            memory_estimation = estimate_memory(
                cfg.model,
                cfg.trainer,
                parallel_config,
                token_imbalance_hypothesis=cfg.model.moe.token_imbalance_hypothesis,
            )
            csv_writer.writerow(memory_estimation)


if __name__ == "__main__":
    main()
