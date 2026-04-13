import re
import math
import statistics
import numpy as np
from typing import Any, Optional, Tuple
from sklearn.metrics import roc_auc_score
from mathruler.grader import extract_boxed_content, grade_answer


def calculate_batch_soft_labels(
    values: list[Optional[float]],
    metric_type: str = "combined",
    entropy_values: Optional[list[Optional[float]]] = None,
    kl_values: Optional[list[Optional[float]]] = None
) -> list[Optional[float]]:
    
    result = [None] * len(values)

    if metric_type == "combined":
        if entropy_values is None or kl_values is None:
            raise ValueError("entropy_values and kl_values are required for metric_type='combined'")

        valid_entries = []
        for i in range(len(values)):
            ent = entropy_values[i]
            kl = kl_values[i]
            if ent is not None and kl is not None:
                valid_entries.append(i)

        num_valid = len(valid_entries)
        if num_valid == 0:
            return result

        epsilon = 1e-8
        combined_scores = []
        for idx in valid_entries:
            kl = kl_values[idx]
            ent = entropy_values[idx]
            score = math.log(kl + epsilon) - math.log(ent + epsilon)
            combined_scores.append(score)

        mean_val = sum(combined_scores) / num_valid
        variance = sum((x - mean_val) ** 2 for x in combined_scores) / num_valid
        std_val = math.sqrt(variance)

        if std_val < 1e-6:
            normalized_scores = [0.0] * num_valid
        else:
            normalized_scores = [(x - mean_val) / std_val for x in combined_scores]

        soft_labels = [1.0 / (1.0 + math.exp(-score)) for score in normalized_scores]

        for i, idx in enumerate(valid_entries):
            result[idx] = soft_labels[i]

        return result

    valid_entries = [(i, val) for i, val in enumerate(values) if val is not None]

    num_valid = len(valid_entries)
    if num_valid == 0:
        return result

    if metric_type == "entropy":
        valid_entries.sort(key=lambda x: x[1])
    elif metric_type == "kl":
        valid_entries.sort(key=lambda x: x[1], reverse=True)
    else:
        raise ValueError(f"Unknown metric_type: {metric_type}")

    if num_valid == 1:
        result[valid_entries[0][0]] = 1.0
        return result

    for rank, (original_idx, _) in enumerate(valid_entries):
        score = 1.0 - (rank / (num_valid - 1))
        result[original_idx] = float(score)

    return result


def compute_vision_confidence_loss(
    extracted_confidence: Optional[float],
    soft_label: Optional[float],
    no_vision_penalty: float = 1.0,
) -> float:
    
    if soft_label is None:
        return -no_vision_penalty
    
    if extracted_confidence is None:
        return -no_vision_penalty
    
    confidence_loss = -((extracted_confidence - soft_label) ** 2)
    
    calibration_bonus = 0.0
    
    return confidence_loss + calibration_bonus


def compute_vision_content_ratio(response: str) -> float:
    
    vision_match = re.search(r'<vision>(.*?)</vision>', response, re.DOTALL)
    
    if not vision_match:
        return 0.0
    
    vision_content = vision_match.group(1).strip()
    
    cleaned_response = re.sub(r'<[^>]+>', '', response)
    
    cleaned_vision_content = re.sub(r'<[^>]+>', '', vision_content)
    
    if len(cleaned_response) == 0:
        return 0.0
    
    ratio = len(cleaned_vision_content) / len(cleaned_response)
    
    return min(1.0, ratio)


def extract_confidence(response: str) -> Optional[tuple[float, float, float]]:
    
    pattern = re.compile(
        r"<confidence>.*?<vision_confidence>(\d+(?:\.\d+)?)</vision_confidence>.*?<reasoning_confidence>(\d+(?:\.\d+)?)</reasoning_confidence>.*?</confidence>",
        re.DOTALL
    )
    match = pattern.search(response)
    
    if match:
        try:
            v_val = float(match.group(1))
            r_val = float(match.group(2))
            
            if (0.0 <= v_val <= 10.0) and (0.0 <= r_val <= 10.0):
                v_norm = v_val / 10.0
                r_norm = r_val / 10.0
                if v_norm == 0.0 or r_norm == 0.0:
                    avg_norm = 0.0
                else:
                    avg_norm = 2.0 / (1.0 / v_norm + 1.0 / r_norm)
                return (avg_norm, v_norm, r_norm)
            return None 
        except ValueError:
            return None
    return None

def format_reward(response: str) -> float:
    
    response = re.sub(r"\s*(<|>|/)\s*", r"\1", response)
    
    pattern = re.compile(
        r"<think>.*?<vision>.*?</vision>.*?<reasoning>.*?</reasoning>.*?</think>.*?\\boxed\{.*?\}.*?<analysis>.*?</analysis>.*?<confidence>.*?<vision_confidence>(.*?)</vision_confidence>.*?<reasoning_confidence>(.*?)</reasoning_confidence>.*?</confidence>",
        re.DOTALL,
    )
    match = pattern.search(response)
    
    if not match:
        return 0.0

    try:
        v_val = float(match.group(1).strip())
        r_val = float(match.group(2).strip())
        if (0.0 <= v_val <= 10.0) and (0.0 <= r_val <= 10.0):
            return 1.0
        return 0.0
    except ValueError:
        return 0.0

def accuracy_reward(response: str, ground_truth: str) -> float:
    
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0

def compute_score(
    reward_inputs: list[dict[str, Any]],
    brier_weight: float = 2.0,
    format_weight: float = 0.4,
    vision_entropy_weight: float = 0.0,
    vision_calibration_weight: float = 0.4,
    vision_kl_weight: float = 0.0,
    soft_label_source: str = "combined",
) -> list[dict[str, float]]:
    
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")
    
    raw_entropies = [inp.get("vision_entropy") for inp in reward_inputs]
    
    raw_vision_kl = [inp.get("vision_kl") for inp in reward_inputs]
    
    has_vision_entropy = any(e is not None for e in raw_entropies)
    has_vision_kl = any(kl is not None for kl in raw_vision_kl)
    
    print(f"\n[DEBUG] compute_score called with {len(reward_inputs)} inputs")
    print(f"[DEBUG] Has vision_entropy in inputs: {has_vision_entropy}")
    print(f"[DEBUG] Has vision_kl in inputs: {has_vision_kl}")
    print(f"[DEBUG] Soft label source: {soft_label_source}")
    
    if soft_label_source == "combined" and has_vision_entropy and has_vision_kl:
        batch_soft_labels = calculate_batch_soft_labels(
            values=raw_entropies,
            metric_type="combined",
            entropy_values=raw_entropies,
            kl_values=raw_vision_kl
        )
        soft_label_metric_used = "Combined (log(kl) - log(entropy), z-score, sigmoid)"
    elif soft_label_source == "kl" and has_vision_kl:
        batch_soft_labels = calculate_batch_soft_labels(raw_vision_kl, metric_type="kl")
        soft_label_metric_used = "Reverse KL divergence"
    elif soft_label_source == "entropy" and has_vision_entropy:
        batch_soft_labels = calculate_batch_soft_labels(raw_entropies, metric_type="entropy")
        soft_label_metric_used = "entropy"
    else:
        if soft_label_source == "combined" and (has_vision_entropy or has_vision_kl):
            if has_vision_kl and not has_vision_entropy:
                batch_soft_labels = calculate_batch_soft_labels(raw_vision_kl, metric_type="kl")
                soft_label_metric_used = "Reverse KL divergence (combined fallback)"
            elif has_vision_entropy and not has_vision_kl:
                batch_soft_labels = calculate_batch_soft_labels(raw_entropies, metric_type="entropy")
                soft_label_metric_used = "entropy (combined fallback)"
            else:
                batch_soft_labels = [None] * len(reward_inputs)
                soft_label_metric_used = "none"
        elif has_vision_kl:
            batch_soft_labels = calculate_batch_soft_labels(raw_vision_kl, metric_type="kl")
            soft_label_metric_used = "Reverse KL divergence (fallback)"
        elif has_vision_entropy:
            batch_soft_labels = calculate_batch_soft_labels(raw_entropies, metric_type="entropy")
            soft_label_metric_used = "entropy (fallback)"
        else:
            batch_soft_labels = [None] * len(reward_inputs)
            soft_label_metric_used = "none"
    
    print(f"[DEBUG] Using {soft_label_metric_used} for soft labels")

    scores = []
    
    valid_c_list = []
    valid_vision_c_list = []
    valid_reasoning_c_list = []
    valid_I_list = []
    valid_brier_list = []
    
    vision_ratio_list = []
    vision_entropy_list = []
    vision_soft_labels_stats = []
    vision_calibration_losses = []
    vision_token_count_list = []
    vision_kl_list = []
    vision_kl_token_count_list = []
    
    all_soft_labels_for_auroc = []
    all_accuracies_for_auroc = []
    
    processed_samples = []

    for idx, reward_input in enumerate(reward_inputs):
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
        gt = reward_input["ground_truth"]
        
        soft_label = batch_soft_labels[idx]
        vision_entropy = raw_entropies[idx]
        vision_kl = raw_vision_kl[idx]
        
        extracted = extract_confidence(response)
        I = accuracy_reward(response, gt)
        fmt = format_reward(response)
        
        vision_ratio = compute_vision_content_ratio(response)
        vision_ratio_list.append(vision_ratio)
        
        if vision_entropy is not None:
            vision_entropy_list.append(vision_entropy)
        
        if soft_label is not None:
            vision_soft_labels_stats.append(soft_label)
        
        vision_token_count = reward_input.get("vision_token_count", None)
        if vision_token_count is not None:
            vision_token_count_list.append(vision_token_count)
        
        if vision_kl is not None:
            vision_kl_list.append(vision_kl)
        
        vision_kl_token_count = reward_input.get("vision_kl_token_count", None)
        if vision_kl_token_count is not None:
            vision_kl_token_count_list.append(vision_kl_token_count)
        
        vision_conf_extracted = None
        if extracted is not None:
            _, vision_conf_extracted, _ = extracted
        
        vision_calibration_loss = compute_vision_confidence_loss(
            vision_conf_extracted,
            soft_label,
            no_vision_penalty=1.0
        )
        vision_calibration_losses.append(vision_calibration_loss)
        
        if len(processed_samples) == 0 and (vision_entropy is not None or vision_kl is not None):
            print("\n" + "="*80)
            print(f"Vision Calibration Debug (First Sample - Rank Based using {soft_label_metric_used}):")
            print("="*80)
            if vision_entropy is not None:
                print(f"Raw Vision Entropy: {vision_entropy:.4f}")
            if vision_kl is not None:
                print(f"Raw Vision Reverse KL: {vision_kl:.6f}")
            if soft_label is not None:
                print(f"Rank-Based Soft Label (Target): {soft_label:.4f}")
            if vision_conf_extracted is not None:
                print(f"Extracted Confidence: {vision_conf_extracted:.4f}")
                if soft_label is not None:
                    print(f"Error (Diff): {abs(vision_conf_extracted - soft_label):.4f}")
            else:
                print(f"Extracted Confidence: None (missing)")
            print(f"Calibration Loss: {vision_calibration_loss:.4f}")
            print("="*80 + "\n")
        
        if extracted is not None:
            c_avg, c_vis, c_rea = extracted
            brier_score = -(c_avg - I) ** 2
            
            valid_c_list.append(c_avg)
            valid_vision_c_list.append(c_vis)
            valid_reasoning_c_list.append(c_rea)
            valid_I_list.append(I)
            valid_brier_list.append(brier_score)
            c_final = c_avg
        else:
            c_final = 0.0
            brier_score = -1.0 

        all_accuracies_for_auroc.append(I)
        all_soft_labels_for_auroc.append(soft_label)

        processed_samples.append({
            "I": I,
            "c": c_final,
            "brier_score": brier_score,
            "fmt": fmt,
            "vision_ratio": vision_ratio,
            "vision_entropy": vision_entropy if vision_entropy is not None else 0.0,
            "vision_kl": vision_kl if vision_kl is not None else 0.0,
            "vision_calibration_loss": vision_calibration_loss,
            "soft_label": soft_label
        })

    stats = {
        "valid_rate": 0.0,
        "brier": float("nan"),
        "variance": 0.0, "std": 0.0, "mean": 0.0, "max": 0.0, "min": 0.0,
        "auroc": float("nan"), "ece": 0.0,
        "soft_label_accuracy_auroc": float("nan"),
        "mean_correct": float("nan"), "mean_wrong": float("nan"),
        "gap": float("nan"), "cohens_d": float("nan"),
        
        "vision_mean": 0.0, "vision_std": 0.0, "vision_max": 0.0, "vision_min": 0.0,
        "reasoning_mean": 0.0, "reasoning_std": 0.0, "reasoning_max": 0.0, "reasoning_min": 0.0,
        
        "vision_ratio_mean": 0.0, "vision_ratio_std": 0.0, 
        "vision_ratio_max": 0.0, "vision_ratio_min": 0.0,
        
        "vision_entropy_mean": 0.0, "vision_entropy_std": 0.0,
        "vision_entropy_max": 0.0, "vision_entropy_min": 0.0,
        "vision_token_count_mean": 0.0,
        
        "vision_kl_mean": 0.0, "vision_kl_std": 0.0,
        "vision_kl_max": 0.0, "vision_kl_min": 0.0,
        "vision_kl_token_count_mean": 0.0,
        
        "vision_calibration_loss_mean": 0.0,
        "vision_soft_label_mean": 0.0,
    }
    
    if vision_soft_labels_stats:
        print("\n" + "="*80)
        print(f"Vision Calibration Summary (Rank Based using {soft_label_metric_used}):")
        print("="*80)
        
        if soft_label_metric_used.startswith("Combined"):
            if vision_entropy_list:
                print(f"Samples with vision entropy: {len(vision_entropy_list)}/{len(reward_inputs)}")
                print(f"Entropy Range: [{min(vision_entropy_list):.4f}, {max(vision_entropy_list):.4f}]")
            if vision_kl_list:
                print(f"Samples with vision Reverse KL: {len(vision_kl_list)}/{len(reward_inputs)}")
                print(f"Reverse KL Range: [{min(vision_kl_list):.6f}, {max(vision_kl_list):.6f}]")
            print(f"Combined soft labels: log(kl) - log(entropy), z-score normalized, then sigmoid to [0, 1]")
        elif soft_label_metric_used.startswith("Reverse KL"):
            if vision_kl_list:
                print(f"Samples with vision Reverse KL: {len(vision_kl_list)}/{len(reward_inputs)}")
                print(f"Reverse KL Range: [{min(vision_kl_list):.6f}, {max(vision_kl_list):.6f}]")
        elif soft_label_metric_used.startswith("entropy"):
            if vision_entropy_list:
                print(f"Samples with vision entropy: {len(vision_entropy_list)}/{len(reward_inputs)}")
                print(f"Entropy Range: [{min(vision_entropy_list):.4f}, {max(vision_entropy_list):.4f}]")
        
        print(f"Soft Labels Distributed in [0, 1] based on {soft_label_metric_used}")
        print(f"Number of samples with soft labels: {len(vision_soft_labels_stats)}")
        
        if vision_calibration_losses:
            print(f"Average calibration loss: {statistics.mean(vision_calibration_losses):.4f}")
        print("="*80 + "\n")
    
    if vision_kl_list:
        print("\n" + "="*80)
        print("Vision Reverse KL Divergence Summary:")
        if soft_label_metric_used.startswith("Reverse KL"):
            print("  [Used for soft label calculation]")
        print("="*80)
        print(f"Samples with vision Reverse KL: {len(vision_kl_list)}/{len(reward_inputs)}")
        print(f"Reverse KL Range: [{min(vision_kl_list):.6f}, {max(vision_kl_list):.6f}]")
        print(f"Reverse KL Mean: {statistics.mean(vision_kl_list):.6f}")
        print(f"Reverse KL Std: {statistics.stdev(vision_kl_list) if len(vision_kl_list) > 1 else 0.0:.6f}")
        if vision_kl_token_count_list:
            print(f"Average Reverse KL token count: {statistics.mean(vision_kl_token_count_list):.1f}")
        print("="*80 + "\n")
    
    if vision_entropy_list and not soft_label_metric_used.startswith("entropy"):
        print("\n" + "="*80)
        print("Vision Entropy Summary:")
        if soft_label_metric_used.startswith("entropy"):
            print("  [Used for soft label calculation]")
        print("="*80)
        print(f"Samples with vision entropy: {len(vision_entropy_list)}/{len(reward_inputs)}")
        print(f"Entropy Range: [{min(vision_entropy_list):.4f}, {max(vision_entropy_list):.4f}]")
        print(f"Entropy Mean: {statistics.mean(vision_entropy_list):.4f}")
        print(f"Entropy Std: {statistics.stdev(vision_entropy_list) if len(vision_entropy_list) > 1 else 0.0:.4f}")
        print("="*80 + "\n")

    total_samples = len(reward_inputs)
    if total_samples > 0:
        stats["valid_rate"] = len(valid_c_list) / total_samples

    if valid_c_list:
        stats["mean"] = statistics.mean(valid_c_list)
        stats["max"] = max(valid_c_list)
        stats["min"] = min(valid_c_list)
        stats["brier"] = statistics.mean(valid_brier_list)
        if len(valid_c_list) > 1:
            stats["variance"] = statistics.variance(valid_c_list)
            stats["std"] = statistics.pstdev(valid_c_list)

        stats["vision_mean"] = statistics.mean(valid_vision_c_list)
        stats["vision_max"] = max(valid_vision_c_list)
        stats["vision_min"] = min(valid_vision_c_list)
        if len(valid_vision_c_list) > 1:
            stats["vision_std"] = statistics.pstdev(valid_vision_c_list)

        stats["reasoning_mean"] = statistics.mean(valid_reasoning_c_list)
        stats["reasoning_max"] = max(valid_reasoning_c_list)
        stats["reasoning_min"] = min(valid_reasoning_c_list)
        if len(valid_reasoning_c_list) > 1:
            stats["reasoning_std"] = statistics.pstdev(valid_reasoning_c_list)

        if vision_ratio_list:
            stats["vision_ratio_mean"] = statistics.mean(vision_ratio_list)
            stats["vision_ratio_max"] = max(vision_ratio_list)
            stats["vision_ratio_min"] = min(vision_ratio_list)
            if len(vision_ratio_list) > 1:
                stats["vision_ratio_std"] = statistics.pstdev(vision_ratio_list)
        
        if vision_entropy_list:
            stats["vision_entropy_mean"] = statistics.mean(vision_entropy_list)
            stats["vision_entropy_max"] = max(vision_entropy_list)
            stats["vision_entropy_min"] = min(vision_entropy_list)
            if len(vision_entropy_list) > 1:
                stats["vision_entropy_std"] = statistics.pstdev(vision_entropy_list)
        
        if vision_token_count_list:
            stats["vision_token_count_mean"] = statistics.mean(vision_token_count_list)
        
        if vision_kl_list:
            stats["vision_kl_mean"] = statistics.mean(vision_kl_list)
            stats["vision_kl_max"] = max(vision_kl_list)
            stats["vision_kl_min"] = min(vision_kl_list)
            if len(vision_kl_list) > 1:
                stats["vision_kl_std"] = statistics.pstdev(vision_kl_list)
        
        if vision_kl_token_count_list:
            stats["vision_kl_token_count_mean"] = statistics.mean(vision_kl_token_count_list)
        
        if vision_calibration_losses:
            stats["vision_calibration_loss_mean"] = statistics.mean(vision_calibration_losses)
        
        if vision_soft_labels_stats:
            stats["vision_soft_label_mean"] = statistics.mean(vision_soft_labels_stats)

        try:
            if len(set(valid_I_list)) > 1:
                stats["auroc"] = roc_auc_score(valid_I_list, valid_c_list)
        except ValueError:
            pass
        
        try:
            valid_pairs = [(acc, sl) for acc, sl in zip(all_accuracies_for_auroc, all_soft_labels_for_auroc) 
                          if sl is not None]
            if valid_pairs and len(set([acc for acc, _ in valid_pairs])) > 1:
                valid_accs, valid_sls = zip(*valid_pairs)
                stats["soft_label_accuracy_auroc"] = roc_auc_score(valid_accs, valid_sls)
                print(f"\n[AUROC] Soft Label vs Accuracy AUROC: {stats['soft_label_accuracy_auroc']:.4f}")
                print(f"         Based on {len(valid_pairs)} samples with soft labels")
        except ValueError as e:
            print(f"[AUROC] Could not compute Soft Label vs Accuracy AUROC: {e}")
            pass

        def compute_ece_valid(confidences, accuracies, n_bins=10):
            if not confidences: return 0.0
            bins = np.linspace(0, 1, n_bins + 1)
            ece = 0.0
            total = len(confidences)
            for i in range(n_bins):
                start, end = bins[i], bins[i + 1]
                if i == n_bins - 1:
                    idx = [j for j, val in enumerate(confidences) if start <= val <= end]
                else:
                    idx = [j for j, val in enumerate(confidences) if start <= val < end]
                if not idx: continue
                avg_conf = np.mean([confidences[j] for j in idx])
                avg_acc = np.mean([accuracies[j] for j in idx])
                ece += abs(avg_conf - avg_acc) * len(idx) / total
            return ece

        stats["ece"] = compute_ece_valid(valid_c_list, valid_I_list)

        correct_conf = [valid_c_list[i] for i, acc in enumerate(valid_I_list) if acc == 1]
        wrong_conf = [valid_c_list[i] for i, acc in enumerate(valid_I_list) if acc == 0]

        if correct_conf: stats["mean_correct"] = statistics.mean(correct_conf)
        if wrong_conf: stats["mean_wrong"] = statistics.mean(wrong_conf)

        if not math.isnan(stats["mean_correct"]) and not math.isnan(stats["mean_wrong"]):
            stats["gap"] = stats["mean_correct"] - stats["mean_wrong"]

        if len(correct_conf) > 1 and len(wrong_conf) > 1:
            var_c = statistics.variance(correct_conf)
            var_w = statistics.variance(wrong_conf)
            n_c, n_w = len(correct_conf), len(wrong_conf)
            pooled_std = math.sqrt(((n_c - 1) * var_c + (n_w - 1) * var_w) / (n_c + n_w - 2))
            stats["cohens_d"] = (stats["mean_correct"] - stats["mean_wrong"]) / pooled_std if pooled_std > 1e-9 else 0.0

    for sample in processed_samples:
        vision_entropy_bonus = 0.0
        if vision_entropy_weight != 0.0 and sample["vision_entropy"] > 0:
            vision_entropy_bonus = vision_entropy_weight * sample["vision_entropy"]
        
        vision_calibration_bonus = vision_calibration_weight * sample["vision_calibration_loss"]
        
        vision_kl_bonus = 0.0
        if vision_kl_weight != 0.0 and sample["vision_kl"] > 0:
            vision_kl_bonus = -vision_kl_weight * sample["vision_kl"]
        
        overall = sample["I"] + brier_weight * sample["brier_score"] + format_weight * sample["fmt"] + vision_entropy_bonus + vision_calibration_bonus + vision_kl_bonus
        
        scores.append({
            "overall": overall,
            "accuracy": sample["I"],
            "format": sample["fmt"],
            "vision_kl": sample.get("vision_kl", 0.0),
            "vision_entropy": sample.get("vision_entropy", 0.0),
            
            "batch_valid_rate": stats["valid_rate"],
            "batch_brier": stats["brier"],
            
            "batch_conf_mean": stats["mean"],
            "batch_conf_std": stats["std"],
            "batch_conf_max": stats["max"],
            "batch_conf_min": stats["min"],
            "batch_conf_variance": stats["variance"],
            "batch_conf_auroc": stats["auroc"],
            "batch_soft_label_accuracy_auroc": stats["soft_label_accuracy_auroc"],
            "batch_conf_ece": stats["ece"],
            "batch_conf_mean_correct": stats["mean_correct"],
            "batch_conf_mean_wrong": stats["mean_wrong"],
            "batch_conf_gap": stats["gap"],
            "batch_conf_cohens_d": stats["cohens_d"],
            
            "batch_vision_conf_mean": stats["vision_mean"],
            "batch_vision_conf_std": stats["vision_std"],
            "batch_vision_conf_max": stats["vision_max"],
            "batch_vision_conf_min": stats["vision_min"],
            
            "batch_reasoning_conf_mean": stats["reasoning_mean"],
            "batch_reasoning_conf_std": stats["reasoning_std"],
            "batch_reasoning_conf_max": stats["reasoning_max"],
            "batch_reasoning_conf_min": stats["reasoning_min"],
            
            "batch_vision_ratio_mean": stats.get("vision_ratio_mean", 0.0),
            "batch_vision_ratio_std": stats.get("vision_ratio_std", 0.0),
            "batch_vision_ratio_max": stats.get("vision_ratio_max", 0.0),
            "batch_vision_ratio_min": stats.get("vision_ratio_min", 0.0),
            
            "batch_vision_entropy_mean": stats.get("vision_entropy_mean", 0.0),
            "batch_vision_entropy_std": stats.get("vision_entropy_std", 0.0),
            "batch_vision_entropy_max": stats.get("vision_entropy_max", 0.0),
            "batch_vision_entropy_min": stats.get("vision_entropy_min", 0.0),
            "batch_vision_token_count_mean": stats.get("vision_token_count_mean", 0.0),
            
            "batch_vision_kl_mean": stats.get("vision_kl_mean", 0.0),
            "batch_vision_kl_std": stats.get("vision_kl_std", 0.0),
            "batch_vision_kl_max": stats.get("vision_kl_max", 0.0),
            "batch_vision_kl_min": stats.get("vision_kl_min", 0.0),
            "batch_vision_kl_token_count_mean": stats.get("vision_kl_token_count_mean", 0.0),
            
            "batch_vision_calibration_loss_mean": stats.get("vision_calibration_loss_mean", 0.0),
            "batch_vision_soft_label_mean": stats.get("vision_soft_label_mean", 0.0),
        })

    return scores