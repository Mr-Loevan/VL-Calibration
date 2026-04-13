#!/bin/bash

set -x

export ACCELERATE_LOG_LEVEL=info
export HYDRA_FULL_ERROR=1
MODEL_PATH=Qwen3-VL-4B-Instruct
REWARD_FUNCTION=./examples/reward_function/decouple.py:compute_score
FORMAT_PROMPT=./examples/format_prompt/Standard_Decouple.jinja
SUFFIX=avg_soft_kl_4e-1 

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=train.jsonl@train \
    data.val_files=val.jsonl@train \
    data.format_prompt=${FORMAT_PROMPT} \
    algorithm.use_token_shaping=true \
    algorithm.use_on_perception=true \
    algorithm.compute_vision_kl=true \
    algorithm.token_shaping_weight_min=0.9 \
    algorithm.token_shaping_weight_max=1.1 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.reward.reward_function=${REWARD_FUNCTION} \
    trainer.experiment_name=decouple_${SUFFIX} \
    trainer.n_gpus_per_node=8 \
    trainer.save_checkpoint_path=./decouple_4b_${SUFFIX}