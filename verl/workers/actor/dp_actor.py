

import os
from collections import defaultdict
from typing import Any, Optional

import torch
import torch.distributed as dist
from einops import rearrange
from ray.experimental.tqdm_ray import tqdm
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributions import Categorical

from ...protocol import DataProto, batch_collate
from ...trainer.core_algos import average_loss, compute_kl, compute_policy_loss
from ...utils import torch_functional as VF
from ...utils.py_functional import append_to_dict
from ...utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from ...utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from ...utils.extract_vision_entropy import batch_compute_vision_entropy, create_vision_token_mask
from .base import BasePPOActor
from .config import ActorConfig


try:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
except ImportError:
    pass


__all__ = ["DataParallelPPOActor"]


class DataParallelPPOActor(BasePPOActor):
    def __init__(
        self,
        config: ActorConfig,
        actor_module: nn.Module,
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
        tokenizer: Optional[Any] = None,
    ):
        
        super().__init__(config)
        self.rank = int(os.getenv("RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.tokenizer = tokenizer

        if config.use_torch_compile:
            self.log_probs_from_logits = torch.compile(VF.log_probs_from_logits, dynamic=True)
        else:
            self.log_probs_from_logits = VF.log_probs_from_logits

    def _forward_micro_batch(self, micro_batch: dict[str, torch.Tensor], temperature: float) -> dict[str, torch.Tensor]:
        
        input_ids = micro_batch["input_ids"]
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        responses = micro_batch["responses"]
        response_length = responses.size(-1)
        if position_ids.dim() == 3:
            position_ids = position_ids.transpose(0, 1)

        multi_modal_inputs = defaultdict(list)
        if "multi_modal_inputs" in micro_batch:
            multi_modal_inputs = batch_collate(micro_batch["multi_modal_inputs"])
            multi_modal_inputs = {key: torch.cat(value, dim=0) for key, value in multi_modal_inputs.items()}
        else:
            multi_modal_inputs = {}

        if self.config.padding_free:
            input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)

            if position_ids.dim() == 3:
                position_ids_rmpad = (
                    index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                    .transpose(0, 1)
                    .unsqueeze(1)
                )
            else:
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)

            if self.config.ulysses_size > 1:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=self.config.ulysses_size
                )
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled, None, self.config.ulysses_size
                )

            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)

            output = self.actor_module(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids_rmpad,
                **multi_modal_inputs,
                use_cache=False,
            )
            logits_rmpad = output.logits.squeeze(0)
            logits_rmpad.div_(temperature)
            log_probs = self.log_probs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

            if getattr(self.config, 'use_on_entropy', False):
                dist = Categorical(logits=logits_rmpad)
                entropy = dist.entropy()
            else:
                entropy = None

            if self.config.ulysses_size > 1:
                log_probs = gather_outputs_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                if entropy is not None:
                    entropy = gather_outputs_and_unpad(entropy, gather_dim=0, unpad_dim=0, padding_size=pad_size)

            full_log_probs = pad_input(
                hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
            )
            log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]
            
            if entropy is not None:
                full_entropy = pad_input(
                    hidden_states=entropy.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
                )
                entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]
            else:
                entropy = None
        else:
            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **multi_modal_inputs,
                use_cache=False,
            )
            logits: torch.Tensor = output.logits
            logits.div_(temperature)
            logits = logits[:, -response_length - 1 : -1, :]
            log_probs = self.log_probs_from_logits(logits, responses)
            
            if getattr(self.config, 'use_on_entropy', False):
                dist = Categorical(logits=logits)
                entropy = dist.entropy()
            else:
                entropy = None

        return {"log_probs": log_probs, "entropy": entropy}

    def _optimizer_step(self) -> torch.Tensor:
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(self.config.max_grad_norm)
        else:
            grad_norm = nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.max_grad_norm)

        if not torch.isfinite(grad_norm):
            print("Gradient norm is not finite. Skip update.")
        else:
            self.actor_optimizer.step()

        self.actor_optimizer.zero_grad()
        return grad_norm

    @torch.no_grad()
    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        
        self.actor_module.eval()

        temperature = data.meta_info["temperature"]
        select_keys = ["input_ids", "attention_mask", "position_ids", "responses"]
        non_tensor_select_keys = ["multi_modal_inputs"]

        data = data.select(select_keys, non_tensor_select_keys)
        if self.config.dynamic_batching:
            max_token_len = self.config.micro_batch_size_per_device_for_experience * data.batch["input_ids"].size(-1)
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(self.config.micro_batch_size_per_device_for_experience)

        log_probs_lst = []
        entropy_lst = []
        if self.rank == 0:
            micro_batches = tqdm(micro_batches, desc="Compute log probs", position=1)

        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            output = self._forward_micro_batch(model_inputs, temperature=temperature)
            log_probs_lst.append(output["log_probs"])
            if "entropy" in output and output["entropy"] is not None:
                entropy_lst.append(output["entropy"])

        log_probs = torch.concat(log_probs_lst, dim=0)
        
        entropy = None
        if entropy_lst:
            entropy = torch.concat(entropy_lst, dim=0)

        if self.config.dynamic_batching:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if entropy is not None:
                entropy = restore_dynamic_batch(entropy, batch_idx_list)

        if self.tokenizer is not None:
            if self.rank == 0:
                print(f"[TRACE] compute_log_prob: Computing vision entropy for batch size: {data.batch['responses'].shape[0]}")
            response_ids = data.batch["responses"]
            response_masks = data.batch.get("response_mask", None)
            
            vision_results = batch_compute_vision_entropy(
                log_probs,
                response_ids,
                self.tokenizer,
                response_masks,
                entropy,
                debug_samples=2 if self.rank == 0 else 0
            )
            
            vision_entropies = [result[0] for result in vision_results]
            vision_token_counts = [result[1] for result in vision_results]
            
            data.non_tensor_batch["vision_entropy"] = vision_entropies
            data.non_tensor_batch["vision_token_count"] = vision_token_counts
            
            valid_count = sum(1 for e in vision_entropies if e is not None)
            if self.rank == 0:
                print(f"[DEBUG] Stored vision entropy: {valid_count}/{len(vision_entropies)} valid samples")
                print(f"[DEBUG] data.non_tensor_batch keys after adding: {data.non_tensor_batch.keys()}")

        if self.tokenizer is not None and "vision_entropy" in data.non_tensor_batch:
            return log_probs, data.non_tensor_batch["vision_entropy"], data.non_tensor_batch.get("vision_token_count")
        return log_probs

    def _compute_token_shaping_weights(
        self,
        kl: torch.Tensor,
        entropy: torch.Tensor,
        response_mask: torch.Tensor,
        weight_min: float = 0.9,
        weight_max: float = 1.1
    ) -> torch.Tensor:
        
        batch_size, seq_len = kl.shape
        device = kl.device
        epsilon = 1e-8

        weights = torch.ones_like(kl)

        for i in range(batch_size):
            valid_mask = response_mask[i].bool()
            if not valid_mask.any():
                continue

            valid_kl = kl[i][valid_mask]
            valid_entropy = entropy[i][valid_mask]

            log_ratio = torch.log(valid_kl + epsilon) - torch.log(valid_entropy + epsilon)

            mean_val = log_ratio.mean()
            std_val = log_ratio.std()

            if std_val < 1e-6:
                normalized = torch.zeros_like(log_ratio)
            else:
                normalized = (log_ratio - mean_val) / std_val

            combined_metric = torch.sigmoid(normalized)

            sample_weights = weight_max - (weight_max - weight_min) * combined_metric

            weights[i][valid_mask] = sample_weights

        return weights

    def update_policy(self, data: DataProto) -> dict[str, Any]:
        self.actor_module.train()

        temperature = data.meta_info["temperature"]
        select_keys = ["input_ids", "attention_mask", "position_ids", "responses", "response_mask"]
        select_keys.extend(["old_log_probs", "ref_log_probs", "advantages"])
        non_tensor_select_keys = ["multi_modal_inputs"]
        
        if self.config.use_on_perception:
            select_keys.append("aug_log_probs")

        mini_batches = data.select(select_keys, non_tensor_select_keys).split(self.config.global_batch_size_per_device)

        metrics = defaultdict(list)
        for _ in range(self.config.ppo_epochs):
            if self.rank == 0:
                mini_batches = tqdm(mini_batches, desc="Train mini-batches", position=1)

            for mini_batch in mini_batches:
                total_response_tokens = torch.sum(mini_batch.batch["response_mask"])
                dist.all_reduce(total_response_tokens, op=dist.ReduceOp.SUM)
                
                global_min_score, global_max_score = None, None
                
                if self.config.use_advantage_shaping and self.config.use_on_perception:
                    with torch.no_grad():
                        mb_old_log_probs = mini_batch.batch["old_log_probs"]
                        mb_aug_log_probs = mini_batch.batch["aug_log_probs"]
                        mb_response_mask = mini_batch.batch["response_mask"]
                        
                        mb_log_probs_diff = (mb_aug_log_probs - mb_old_log_probs).clamp(-20.0, 20.0)
                        mb_low_var_kl = (mb_log_probs_diff.exp() - mb_log_probs_diff - 1).contiguous()
                        mb_low_var_kl = torch.clamp(mb_low_var_kl, min=0.0, max=10.0)
                        
                        mb_num_valid_tokens = mb_response_mask.sum(dim=1)
                        mb_num_valid_tokens_safe = torch.clamp(mb_num_valid_tokens, min=1)
                        
                        mb_masked_low_var_kl = mb_low_var_kl * mb_response_mask
                        mb_sum_low_var_kl = mb_masked_low_var_kl.sum(dim=1)
                        
                        all_sensitivity_scores = mb_sum_low_var_kl / mb_num_valid_tokens_safe
                        
                        valid_scores_mask = mb_num_valid_tokens > 0
                        valid_scores = all_sensitivity_scores[valid_scores_mask]
                        
                        if valid_scores.numel() > 1:
                            global_min_score = torch.quantile(valid_scores, 0.0)
                            global_max_score = torch.quantile(valid_scores, 1.0)

                if self.config.dynamic_batching:
                    max_input_len = mini_batch.batch["input_ids"].size(-1)
                    max_token_len = self.config.micro_batch_size_per_device_for_update * max_input_len
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    micro_batches = mini_batch.split(self.config.micro_batch_size_per_device_for_update)

                if self.rank == 0:
                    micro_batches = tqdm(micro_batches, desc="Update policy", position=2)

                for micro_batch in micro_batches:
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_probs = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]
                    response_ids = model_inputs.get("responses", None)

                    output = self._forward_micro_batch(model_inputs, temperature=temperature)
                    log_probs = output["log_probs"]
                    entropy = output["entropy"]
                    
                    loss_token_mask = None
                    
                    if self.config.use_on_entropy:
                        top_p = self.config.top_p_entropy_tokens
                        num_valid_tokens = response_mask.sum(dim=1)
                        k = torch.ceil(num_valid_tokens * top_p).int()
                        
                        masked_entropy = entropy.clone()
                        masked_entropy[~response_mask.bool()] = -float('inf')
                        sorted_entropy_vals, sorted_indices = torch.sort(masked_entropy, dim=1, descending=True)
                        
                        range_tensor = torch.arange(entropy.size(1), device=entropy.device).expand_as(entropy)
                        rank_mask = range_tensor < k.unsqueeze(1)
                        top_p_mask = torch.zeros_like(entropy, dtype=torch.bool)
                        top_p_mask.scatter_(1, sorted_indices, rank_mask)
                        loss_token_mask = top_p_mask.to(entropy.dtype)
                        
                        with torch.no_grad():
                            if response_mask.sum() > 0:
                                metrics["actor/entropy_token_fraction"].append(
                                    (top_p_mask.sum() / response_mask.sum()).item()
                                )
                    
                    if self.config.use_on_perception:
                        top_p = self.config.top_p_perception_tokens
                        aug_log_probs = model_inputs["aug_log_probs"]
                        
                        log_probs_diff = (aug_log_probs - old_log_probs).clamp(-20.0, 20.0)
                        low_var_kl = (log_probs_diff.exp() - log_probs_diff - 1).contiguous()
                        low_var_kl = torch.clamp(low_var_kl, min=0.0, max=10.0)
                        
                        low_var_kl_for_sort = low_var_kl.clone()
                        low_var_kl_for_sort[~response_mask.bool()] = -torch.inf
                        
                        num_valid_tokens = response_mask.sum(dim=1)
                        k = torch.ceil(num_valid_tokens * top_p).int()
                        
                        sorted_vals, sorted_indices = torch.sort(low_var_kl_for_sort, dim=1, descending=True)
                        range_tensor = torch.arange(low_var_kl.size(1), device=low_var_kl.device).expand_as(low_var_kl)
                        rank_mask = range_tensor < k.unsqueeze(1)
                        
                        top_p_mask = torch.zeros_like(low_var_kl, dtype=torch.bool)
                        top_p_mask.scatter_(1, sorted_indices, rank_mask)
                        top_p_mask = (top_p_mask.bool() & response_mask.bool()).to(log_probs.dtype)
                        
                        if loss_token_mask is not None:
                            loss_token_mask = (loss_token_mask.bool() | top_p_mask.bool()).to(log_probs.dtype)
                        else:
                            loss_token_mask = top_p_mask
                        
                        with torch.no_grad():
                            if response_mask.sum() > 0:
                                metrics["actor/perception_token_fraction"].append(
                                    (top_p_mask.sum() / response_mask.sum()).item()
                                )

                    if self.config.use_token_shaping:
                        with torch.no_grad():
                            if self.config.use_on_perception and "aug_log_probs" in model_inputs:
                                aug_log_probs = model_inputs["aug_log_probs"]
                                log_probs_diff = (aug_log_probs - old_log_probs).clamp(-20.0, 20.0)
                                token_kl = (log_probs_diff.exp() - log_probs_diff - 1).contiguous()
                                token_kl = torch.clamp(token_kl, min=0.0, max=10.0)
                            else:
                                token_kl = -log_probs

                            if entropy is None:
                                token_entropy = -log_probs
                            else:
                                token_entropy = entropy

                            token_weights = self._compute_token_shaping_weights(
                                kl=token_kl,
                                entropy=token_entropy,
                                response_mask=response_mask,
                                weight_min=self.config.token_shaping_weight_min,
                                weight_max=self.config.token_shaping_weight_max
                            )

                            if response_ids is not None and self.tokenizer is not None:
                                vision_token_mask = create_vision_token_mask(
                                    response_ids=response_ids,
                                    tokenizer=self.tokenizer,
                                    response_mask=response_mask,
                                    debug=False
                                )

                                final_weights = torch.ones_like(token_weights)
                                final_weights[vision_token_mask] = token_weights[vision_token_mask]

                                vision_weights = token_weights[vision_token_mask]
                                if vision_weights.numel() > 0:
                                    metrics["actor/token_shaping_vision_weight_mean"].append(vision_weights.mean().item())
                                    metrics["actor/token_shaping_vision_weight_min"].append(vision_weights.min().item())
                                    metrics["actor/token_shaping_vision_weight_max"].append(vision_weights.max().item())

                                    metrics["actor/vision_token_fraction"].append(
                                        vision_token_mask.sum().float() / response_mask.sum().float()
                                    )
                            else:
                                final_weights = token_weights
                                metrics["actor/token_shaping_weight_mean"].append(token_weights.mean().item())

                            negative_adv_mask = (advantages < 0).float()
                            effective_weights = negative_adv_mask * final_weights + (1 - negative_adv_mask) * 1.0
                            advantages = advantages * effective_weights

                    if self.config.use_advantage_shaping and 'low_var_kl' in locals():
                        with torch.no_grad():
                            num_valid_tokens = response_mask.sum(dim=1)
                            num_valid_tokens_safe = torch.clamp(num_valid_tokens, min=1)
                            masked_low_var_kl = low_var_kl * response_mask
                            sum_low_var_kl = masked_low_var_kl.sum(dim=1)
                            sensitivity_score = sum_low_var_kl / num_valid_tokens_safe
                            
                            scaling_factor = torch.ones_like(sensitivity_score)
                            if global_min_score is not None and global_max_score is not None:
                                if (global_max_score - global_min_score) > 1e-6:
                                    valid_scores_mask = num_valid_tokens > 0
                                    valid_scores = sensitivity_score[valid_scores_mask]
                                    normalized_scores = (valid_scores - global_min_score) / (global_max_score - global_min_score)
                                    normalized_scores = torch.clamp(normalized_scores, 0.0, 1.0)
                                    
                                    target_min = self.config.advantage_scaling_min
                                    mu_norm = normalized_scores.mean()
                                    target_max = target_min + (1.0 - target_min) / (mu_norm + 1e-8)
                                    target_range = target_max - target_min
                                    mapped_scores = target_min + normalized_scores * target_range
                                    scaling_factor[valid_scores_mask] = mapped_scores
                                    
                                    metrics["actor/scaling_factor_mean"].append(scaling_factor.mean().item())
                        
                        advantages = advantages * scaling_factor.unsqueeze(1)
                    
                    pg_loss, pg_metrics = compute_policy_loss(
                        old_log_probs=old_log_probs,
                        log_probs=log_probs,
                        advantages=advantages,
                        response_mask=response_mask,
                        clip_ratio_low=self.config.clip_ratio_low,
                        clip_ratio_high=self.config.clip_ratio_high,
                        clip_ratio_dual=self.config.clip_ratio_dual,
                        loss_avg_mode=self.config.loss_avg_mode,
                        loss_token_mask=loss_token_mask,
                    )
                    if self.config.use_kl_loss and "ref_log_probs" in model_inputs:
                        ref_log_probs = model_inputs["ref_log_probs"]
                        kld = compute_kl(
                            log_probs=log_probs,
                            ref_log_probs=ref_log_probs,
                            kl_penalty=self.config.kl_penalty,
                        )
                        kl_loss = average_loss(kld, response_mask, mode=self.config.loss_avg_mode)
                        loss = pg_loss + kl_loss * self.config.kl_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_coef
                    else:
                        loss = pg_loss

                    loss = loss * torch.sum(response_mask) * self.world_size / total_response_tokens
                    loss.backward()

                    if self.config.use_entropy_penalty:
                        entropy_loss = -VF.masked_mean(log_probs, response_mask)
                        loss = loss + entropy_loss * self.config.entropy_penalty_coef
                        metrics["actor/entropy_penalty_coef"] = self.config.entropy_penalty_coef
                    
                    batch_metrics = {f"actor/{k}": v for k, v in pg_metrics.items()}
                    batch_metrics["actor/pg_loss"] = pg_loss.detach().item()
                    append_to_dict(metrics, batch_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        return metrics
