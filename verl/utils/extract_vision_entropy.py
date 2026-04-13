

from typing import List, Optional, Tuple

import torch


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


def create_vision_token_mask(
    response_ids: torch.Tensor,
    tokenizer,
    response_mask: Optional[torch.Tensor] = None,
    vision_start_str: str = "<vision>",
    vision_end_str: str = "</vision>",
    debug: bool = False
) -> torch.Tensor:
    
    batch_size, seq_len = response_ids.shape
    device = response_ids.device
    vision_token_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)

    for i in range(batch_size):
        response_ids_i = response_ids[i]
        start_idx, end_idx = find_vision_boundaries_robust(
            response_ids_i,
            tokenizer,
            vision_start_str,
            vision_end_str,
            debug=debug and i < 2
        )

        if start_idx is not None and end_idx is not None:
            if start_idx < seq_len and end_idx <= seq_len:
                vision_token_mask[i, start_idx:end_idx] = True

    if response_mask is not None:
        vision_token_mask = vision_token_mask & response_mask.bool()

    return vision_token_mask


def compute_vision_entropy_with_debug(
    log_probs: torch.Tensor,
    response_ids: torch.Tensor,
    tokenizer,
    response_mask: Optional[torch.Tensor] = None,
    entropy: Optional[torch.Tensor] = None,
    vision_start_str: str = "<vision>",
    vision_end_str: str = "</vision>",
    debug: bool = False,
    sample_idx: int = 0
) -> Tuple[Optional[float], Optional[int]]:
    
    if response_mask is not None:
        valid_length = int(response_mask.sum().item())
        log_probs = log_probs[:valid_length]
        response_ids = response_ids[:valid_length]
        if entropy is not None:
            entropy = entropy[:valid_length]
    
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
    vision_log_probs = log_probs[start_idx:end_idx]
    num_vision_tokens = end_idx - start_idx
    
    if num_vision_tokens <= 0:
        return None, None
    
    if entropy is not None:
        vision_entropies = entropy[start_idx:end_idx]
    else:
        vision_entropies = -vision_log_probs
    
    vision_entropy_mean = vision_entropies.mean().item()
    
    if debug:
        vision_text = tokenizer.decode(vision_token_ids, skip_special_tokens=False)
        
        display_count = min(10, num_vision_tokens)
        
        print(f"\n{'='*80}")
        print(f"[Sample {sample_idx}] Vision Entropy Analysis")
        print(f"{'='*80}")
        print(f"Vision token range: [{start_idx}:{end_idx}] ({num_vision_tokens} tokens)")
        print(f"Vision text: {vision_text[:200]}{'...' if len(vision_text) > 200 else ''}")
        print(f"\nFirst {display_count} vision tokens:")
        
        for i in range(display_count):
            token_id = vision_token_ids[i].item()
            token_str = tokenizer.decode([token_id], skip_special_tokens=False)
            log_prob = vision_log_probs[i].item()
            entropy = -log_prob
            print(f"  Token {i:3d}: ID={token_id:6d}, Text={repr(token_str):20s}, "
                  f"LogProb={log_prob:8.4f}, Entropy={entropy:8.4f}")
        
        if num_vision_tokens > display_count:
            print(f"  ... ({num_vision_tokens - display_count} more tokens)")
        
        print(f"\nVision Entropy Statistics:")
        print(f"  Mean entropy: {vision_entropy_mean:.4f}")
        print(f"  Min entropy:  {vision_entropies.min().item():.4f}")
        print(f"  Max entropy:  {vision_entropies.max().item():.4f}")
        print(f"  Std entropy:  {vision_entropies.std().item():.4f}")
        print("="*80 + "\n")
    
    return vision_entropy_mean, num_vision_tokens


def batch_compute_vision_entropy(
    log_probs_batch: torch.Tensor,
    response_ids_batch: torch.Tensor,
    tokenizer,
    response_masks: Optional[torch.Tensor] = None,
    entropy_batch: Optional[torch.Tensor] = None,
    vision_start_str: str = "<vision>",
    vision_end_str: str = "</vision>",
    debug_samples: int = 0
) -> List[Tuple[Optional[float], Optional[int]]]:
    
    batch_size = log_probs_batch.shape[0]
    results = []
    
    valid_entropies = []
    valid_token_counts = []
    
    for i in range(batch_size):
        response_mask = response_masks[i] if response_masks is not None else None
        entropy_tensor = entropy_batch[i] if entropy_batch is not None else None
        
        debug_this_sample = i < debug_samples
        
        entropy, num_tokens = compute_vision_entropy_with_debug(
            log_probs_batch[i],
            response_ids_batch[i],
            tokenizer,
            response_mask,
            entropy_tensor,
            vision_start_str,
            vision_end_str,
            debug=debug_this_sample,
            sample_idx=i
        )
        
        results.append((entropy, num_tokens))
        
        if entropy is not None:
            valid_entropies.append(entropy)
            if num_tokens is not None:
                valid_token_counts.append(num_tokens)
    
    if debug_samples > 0 and valid_entropies:
        print("\nBatch Summary:")
        print(f"  Total samples: {batch_size}")
        print(f"  Samples with vision: {len(valid_entropies)}")
        print(f"  Average vision entropy: {sum(valid_entropies)/len(valid_entropies):.4f}")
        if valid_token_counts:
            print(f"  Average vision tokens: {sum(valid_token_counts)/len(valid_token_counts):.1f}")
        print("="*80 + "\n")
    
    return results