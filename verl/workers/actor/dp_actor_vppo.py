
def update_policy_micro_batch_loop(self, micro_batch, temperature, response_mask, old_log_probs, advantages, 
                                  total_response_tokens, metrics, global_min_score=None, global_max_score=None):
    
    
    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
    
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
            num_total_valid_tokens = response_mask.sum()
            num_selected_tokens = top_p_mask.sum()
            
            if num_total_valid_tokens > 0:
                actual_token_fraction = (num_selected_tokens / num_total_valid_tokens).item()
                metrics["actor/entropy_token_fraction"].append(actual_token_fraction)
                
                k_safe = k.clone().clamp(min=1)
                threshold_indices = (k_safe - 1).unsqueeze(1)
                threshold_per_response = torch.gather(sorted_entropy_vals, 1, threshold_indices.long()).squeeze(1)
                valid_thresholds = threshold_per_response[k > 0]
                if valid_thresholds.numel() > 0:
                    threshold = valid_thresholds.mean()
                    metrics["actor/entropy_threshold"].append(threshold.item())
                
                selected_entropies = torch.masked_select(entropy, top_p_mask.bool())
                if selected_entropies.numel() > 0:
                    metrics["actor/entropy_mean_selected"].append(selected_entropies.mean().item())
                
                rejected_mask = response_mask.bool() & ~top_p_mask.bool()
                rejected_entropies = torch.masked_select(entropy, rejected_mask)
                if rejected_entropies.numel() > 0:
                    metrics["actor/entropy_mean_rejected"].append(rejected_entropies.mean().item())
    
    if self.config.use_on_perception:
        top_p = self.config.top_p_perception_tokens
        
        aug_log_probs = model_inputs["aug_log_probs"]
        log_probs_diff = (aug_log_probs - old_log_probs).clamp(-20.0, 20.0)
        low_var_kl = (log_probs_diff.exp() - log_probs_diff - 1).contiguous()
        low_var_kl = torch.clamp(low_var_kl, min=0.0, max=10.0)
        
        low_var_kl_for_sort = low_var_kl.clone()
        invalid_mask = ~response_mask.bool()
        low_var_kl_for_sort[invalid_mask] = -torch.inf
        
        num_valid_tokens = response_mask.sum(dim=1)
        k = torch.ceil(num_valid_tokens * top_p).int()
        
        sorted_vals, sorted_indices = torch.sort(low_var_kl_for_sort, dim=1, descending=True)
        
        range_tensor = torch.arange(low_var_kl_for_sort.size(1), device=low_var_kl_for_sort.device).expand_as(low_var_kl_for_sort)
        rank_mask = range_tensor < k.unsqueeze(1)
        
        top_p_mask = torch.zeros_like(low_var_kl_for_sort, dtype=torch.bool)
        top_p_mask.scatter_(1, sorted_indices, rank_mask)
        
        top_p_mask = (top_p_mask.bool() & response_mask.bool()).to(log_probs.dtype)
        
        if loss_token_mask is not None:
            loss_token_mask = (loss_token_mask.bool() | top_p_mask.bool()).to(log_probs.dtype)
        else:
            loss_token_mask = top_p_mask
        
        with torch.no_grad():
            num_total_valid_tokens = response_mask.sum()
            num_selected_tokens = top_p_mask.sum()
            
            if num_total_valid_tokens > 0:
                actual_token_fraction = (num_selected_tokens / num_total_valid_tokens).item()
                metrics["actor/perception_token_fraction"].append(actual_token_fraction)
                
                k_safe = k.clone().clamp(min=1)
                threshold_indices = (k_safe - 1).unsqueeze(1)
                threshold_per_response = torch.gather(sorted_vals, 1, threshold_indices.long()).squeeze(1)
                valid_thresholds = threshold_per_response[k > 0]
                if valid_thresholds.numel() > 0:
                    threshold = valid_thresholds.mean()
                    metrics["actor/low_var_kl_threshold"].append(threshold.item())
                
                selected_low_var_kl = torch.masked_select(low_var_kl, top_p_mask.bool())
                if selected_low_var_kl.numel() > 0:
                    metrics["actor/low_var_kl_mean_selected"].append(selected_low_var_kl.mean().item())
                
                rejected_mask = response_mask.bool() & ~top_p_mask.bool()
                rejected_low_var_kl = torch.masked_select(low_var_kl, rejected_mask)
                if rejected_low_var_kl.numel() > 0:
                    metrics["actor/low_var_kl_mean_rejected"].append(rejected_low_var_kl.mean().item())
    
    if self.config.use_on_entropy and self.config.use_on_perception:
        metrics["actor/combined_token_fraction"].append(
            (loss_token_mask.sum() / response_mask.sum()).item()
        )
    
    if self.config.use_advantage_shaping and 'low_var_kl' in locals():
        with torch.no_grad():
            num_valid_tokens = response_mask.sum(dim=1)
            num_valid_tokens_safe = torch.clamp(num_valid_tokens, min=1)
            
            masked_low_var_kl = low_var_kl * response_mask
            sum_low_var_kl = masked_low_var_kl.sum(dim=1)
            
            sensitivity_score = sum_low_var_kl / num_valid_tokens_safe
            
            scaling_factor = torch.ones_like(sensitivity_score)
            
            if global_min_score is not None and global_max_score is not None and (global_max_score - global_min_score) > 1e-6:
                valid_scores_mask = num_valid_tokens > 0
                valid_scores = sensitivity_score[valid_scores_mask]
                
                normalized_scores = (valid_scores - global_min_score) / (global_max_score - global_min_score)
                normalized_scores = torch.clamp(normalized_scores, 0.0, 1.0)
                
                target_min = self.config.advantage_scaling_min
                
                mu_norm = normalized_scores.mean()
                epsilon = 1e-8
                target_max = target_min + (1.0 - target_min) / (mu_norm + epsilon)
                
                metrics["actor/dynamic_scaling_max"].append(target_max.item())
                
                target_range = target_max - target_min
                mapped_scores = target_min + normalized_scores * target_range
                
                scaling_factor[valid_scores_mask] = mapped_scores
            
            if (num_valid_tokens > 0).any():
                metrics["actor/sensitivity_score_mean"].append(sensitivity_score[num_valid_tokens > 0].mean().item())
            if global_min_score is not None:
                metrics["actor/global_sensitivity_score_min"].append(global_min_score.item())
                metrics["actor/global_sensitivity_score_max"].append(global_max_score.item())
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
    
    if self.config.use_entropy_penalty:
        entropy_loss = -VF.masked_mean(log_probs, response_mask)
        loss = loss + entropy_loss * self.config.entropy_penalty_coef
        metrics["actor/entropy_penalty_coef"] = self.config.entropy_penalty_coef
    
    loss = loss * torch.sum(response_mask) * self.world_size / total_response_tokens
    loss.backward()
    
    batch_metrics = {f"actor/{k}": v for k, v in pg_metrics.items()}
    batch_metrics["actor/pg_loss"] = pg_loss.detach().item()
    append_to_dict(metrics, batch_metrics)
    
    return metrics