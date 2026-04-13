

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


def find_pattern_in_tokens(
    token_ids: List[int],
    pattern_tokens: List[int],
    start_idx: int = 0
) -> Optional[int]:
    
    pattern_len = len(pattern_tokens)
    if pattern_len == 0:
        return None
    
    for i in range(start_idx, len(token_ids) - pattern_len + 1):
        if token_ids[i:i + pattern_len] == pattern_tokens:
            return i
    
    return None


def find_vision_boundaries_robust(
    response_ids: torch.Tensor,
    tokenizer,
    vision_start_str: str = "<vision>",
    vision_end_str: str = "</vision>",
    debug: bool = False
) -> Tuple[Optional[int], Optional[int]]:
    
    response_list = response_ids.tolist()
    
    response_text = tokenizer.decode(response_ids, skip_special_tokens=False)
    
    if debug:
        print(f"DEBUG: First 500 chars of response: {response_text[:500]}")
        print(f"DEBUG: Searching for '{vision_start_str}' and '{vision_end_str}'")
    
    start_text_pos = response_text.find(vision_start_str)
    if start_text_pos == -1:
        if debug:
            print(f"DEBUG: '{vision_start_str}' not found in text")
        return None, None
    
    end_text_pos = response_text.find(vision_end_str, start_text_pos + len(vision_start_str))
    if end_text_pos == -1:
        if debug:
            print(f"DEBUG: '{vision_end_str}' not found in text")
        return None, None
    
    
    text_before_vision = response_text[:start_text_pos + len(vision_start_str)]
    tokens_before_vision = tokenizer.encode(text_before_vision, add_special_tokens=False)
    
    text_until_vision_end = response_text[:end_text_pos]
    tokens_until_vision_end = tokenizer.encode(text_until_vision_end, add_special_tokens=False)
    
    start_idx = len(tokens_before_vision)
    end_idx = len(tokens_until_vision_end)
    
    if start_idx >= end_idx or start_idx >= len(response_list) or end_idx > len(response_list):
        if debug:
            print(f"DEBUG: Method 1 failed (start={start_idx}, end={end_idx}, len={len(response_list)})")
            print(f"DEBUG: Trying Method 2 with pattern variations")
        
        start_variations = [
            vision_start_str + "\n",
            vision_start_str,
            "\n" + vision_start_str + "\n",
            "\n" + vision_start_str,
            " " + vision_start_str + "\n",
            " " + vision_start_str,
            vision_start_str + " ",
        ]
        
        end_variations = [
            "\n" + vision_end_str,
            vision_end_str,
            "\n" + vision_end_str + "\n",
            vision_end_str + "\n",
            " " + vision_end_str,
            vision_end_str + " ",
        ]
        
        start_pos = None
        for variant in start_variations:
            pattern_ids = tokenizer.encode(variant, add_special_tokens=False)
            pos = find_pattern_in_tokens(response_list, pattern_ids)
            if pos is not None:
                start_pos = pos + len(pattern_ids)
                if debug:
                    print(f"DEBUG: Found start with variant '{repr(variant)}' at position {pos}")
                break
        
        if start_pos is None:
            if debug:
                print("DEBUG: No start position found with any variant")
            return None, None
        
        end_pos = None
        for variant in end_variations:
            pattern_ids = tokenizer.encode(variant, add_special_tokens=False)
            pos = find_pattern_in_tokens(response_list, pattern_ids, start_pos)
            if pos is not None:
                end_pos = pos
                if debug:
                    print(f"DEBUG: Found end with variant '{repr(variant)}' at position {pos}")
                break
        
        if end_pos is None:
            if debug:
                print("DEBUG: No end position found with any variant")
            return None, None
        
        start_idx = start_pos
        end_idx = end_pos
    
    if start_idx >= end_idx:
        if debug:
            print(f"DEBUG: Invalid indices (start={start_idx} >= end={end_idx})")
        return None, None
    
    if debug:
        vision_tokens = response_ids[start_idx:end_idx]
        vision_text = tokenizer.decode(vision_tokens, skip_special_tokens=False)
        print(f"DEBUG: Found vision section [{start_idx}:{end_idx}]")
        print(f"DEBUG: Vision text preview: {vision_text[:100]}...")
    
    return start_idx, end_idx


def compute_kl(log_probs_p: torch.Tensor, log_probs_q: torch.Tensor) -> torch.Tensor:
    
    log_probs_diff = (log_probs_p - log_probs_q).clamp(-20.0, 20.0)

    kl_div = log_probs_p.exp() * log_probs_diff

    kl_div = torch.clamp(kl_div, min=0.0, max=10.0)

    return kl_div


def compute_vision_kl_with_debug(
    old_log_probs: torch.Tensor,
    aug_log_probs: torch.Tensor,
    response_ids: torch.Tensor,
    tokenizer,
    response_mask: Optional[torch.Tensor] = None,
    vision_start_str: str = "<vision>",
    vision_end_str: str = "</vision>",
    debug: bool = False,
    sample_idx: int = 0
) -> Tuple[Optional[float], Optional[int]]:
    
    if response_mask is not None:
        valid_length = int(response_mask.sum().item())
        old_log_probs = old_log_probs[:valid_length]
        aug_log_probs = aug_log_probs[:valid_length]
        response_ids = response_ids[:valid_length]
    
    start_idx, end_idx = find_vision_boundaries_robust(
        response_ids, tokenizer, vision_start_str, vision_end_str, 
        debug=debug and sample_idx < 2
    )
    
    if start_idx is None or end_idx is None:
        if debug:
            response_text = tokenizer.decode(response_ids, skip_special_tokens=False)
            print(f"\n[Sample {sample_idx}] No vision section found in response")
            print(f"  Response preview (first 300 chars): {response_text[:300]}...")
            if vision_start_str in response_text:
                print(f"  NOTE: '{vision_start_str}' found in text but couldn't map to tokens!")
                print(f"  First 20 tokens: {response_ids[:20].tolist()}")
                print(f"  Decoded first 20 tokens:")
                for i in range(min(20, len(response_ids))):
                    token_text = tokenizer.decode([response_ids[i].item()], skip_special_tokens=False)
                    print(f"    Token {i}: {response_ids[i].item()} = '{repr(token_text)}'")
        return None, None
    
    vision_token_ids = response_ids[start_idx:end_idx]
    vision_old_log_probs = old_log_probs[start_idx:end_idx]
    vision_aug_log_probs = aug_log_probs[start_idx:end_idx]
    num_vision_tokens = end_idx - start_idx
    
    if num_vision_tokens <= 0:
        return None, None
    
    vision_kl_values = compute_kl(vision_old_log_probs, vision_aug_log_probs)
    
    vision_kl_mean = vision_kl_values.mean().item()
    
    if debug:
        vision_text = tokenizer.decode(vision_token_ids, skip_special_tokens=False)
        
        display_count = min(10, num_vision_tokens)
        
        print(f"\n{'='*80}")
        print(f"[Sample {sample_idx}] Vision K3 Estimator Analysis")
        print(f"{'='*80}")
        print(f"Vision token range: [{start_idx}:{end_idx}] ({num_vision_tokens} tokens)")
        print(f"Vision text: {vision_text[:200]}{'...' if len(vision_text) > 200 else ''}")
        print(f"\nFirst {display_count} vision tokens:")
        
        for i in range(display_count):
            token_id = vision_token_ids[i].item()
            token_str = tokenizer.decode([token_id], skip_special_tokens=False)
            old_log_prob = vision_old_log_probs[i].item()
            aug_log_prob = vision_aug_log_probs[i].item()
            kl_val = vision_kl_values[i].item()
            print(f"  Token {i:3d}: ID={token_id:6d}, Text={repr(token_str):20s}, "
                  f"Old_LogP={old_log_prob:8.4f}, Aug_LogP={aug_log_prob:8.4f}, K3={kl_val:8.6f}")
        
        if num_vision_tokens > display_count:
            print(f"  ... ({num_vision_tokens - display_count} more tokens)")
        
        print(f"\nVision K3 Estimator Statistics:")
        print(f"  Mean K3:  {vision_kl_mean:.6f}")
        print(f"  Min K3:   {vision_kl_values.min().item():.6f}")
        print(f"  Max K3:   {vision_kl_values.max().item():.6f}")
        print(f"  Std K3:   {vision_kl_values.std().item():.6f}")
        print("="*80 + "\n")
    
    return vision_kl_mean, num_vision_tokens


def batch_compute_vision_kl(
    old_log_probs_batch: torch.Tensor,
    aug_log_probs_batch: torch.Tensor,
    response_ids_batch: torch.Tensor,
    tokenizer,
    response_masks: Optional[torch.Tensor] = None,
    vision_start_str: str = "<vision>",
    vision_end_str: str = "</vision>",
    debug_samples: int = 0
) -> List[Tuple[Optional[float], Optional[int]]]:
    
    batch_size = old_log_probs_batch.shape[0]
    results = []
    
    valid_kl_values = []
    valid_token_counts = []
    
    for i in range(batch_size):
        response_mask = response_masks[i] if response_masks is not None else None
        
        debug_this_sample = i < debug_samples
        
        kl, num_tokens = compute_vision_kl_with_debug(
            old_log_probs_batch[i],
            aug_log_probs_batch[i],
            response_ids_batch[i],
            tokenizer,
            response_mask,
            vision_start_str,
            vision_end_str,
            debug=debug_this_sample,
            sample_idx=i
        )
        
        results.append((kl, num_tokens))
        
        if kl is not None:
            valid_kl_values.append(kl)
            if num_tokens is not None:
                valid_token_counts.append(num_tokens)
    
    if debug_samples > 0 and valid_kl_values:
        print("\nBatch Summary:")
        print(f"  Total samples: {batch_size}")
        print(f"  Samples with vision: {len(valid_kl_values)}")
        
        arithmetic_mean = sum(valid_kl_values)/len(valid_kl_values)
        
        print(f"  Average vision K3: {arithmetic_mean:.6f}")
        if valid_token_counts:
            print(f"  Average vision tokens: {sum(valid_token_counts)/len(valid_token_counts):.1f}")
        print("="*80 + "\n")
    
    return results