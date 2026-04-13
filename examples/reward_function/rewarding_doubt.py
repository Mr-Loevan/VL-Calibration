import re
import math
import statistics
from typing import Any
import numpy as np
from sklearn.metrics import roc_auc_score

SCALE = 10.0
MAX_REWARD = -0.0010005003335835344
MIN_REWARD = -6.907755278982137 / 2
WRONG_FORMAT_PENALTY = -SCALE * 3.0

def extract_confidence(response: str) -> float:
    
    try:
        return float(response.strip())
    except ValueError:
        return None

def format_reward(response: str) -> float:
    
    response_clean = response.strip()
    match = re.fullmatch(r"(?:[0-9]|10)(?:\.\d+)?", response_clean)
    return 1.0 if match else 0.0

def compute_ece(confidences, accuracies, n_bins=10):
    
    confidences_norm = [min(0.999, max(0.001, c / 10.0)) for c in confidences]
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        start, end = bins[i], bins[i + 1]
        if i < n_bins - 1:
            idx = [j for j, c in enumerate(confidences_norm) if start <= c < end]
        else:
            idx = [j for j, c in enumerate(confidences_norm) if start <= c <= end]
        if not idx:
            continue
        avg_conf = np.mean([confidences_norm[j] for j in idx])
        avg_acc = np.mean([accuracies[j] for j in idx])
        ece += abs(avg_conf - avg_acc) * len(idx) / len(confidences)
    return ece


def log_based_reward(confidence: float, is_answer_correct: bool) -> float:
    
    if confidence is None or confidence > 10 or confidence < 0:
        return WRONG_FORMAT_PENALTY
    
    normalized_confidence = min(0.999, max(0.001, confidence / 10.0))
    
    if is_answer_correct:
        score = math.log(normalized_confidence)
    else:
        score = math.log(1 - normalized_confidence)
    
    norm_score = (score - MIN_REWARD) / (MAX_REWARD - MIN_REWARD)
    
    return float(SCALE * norm_score)

def compute_score(
    reward_inputs: list[dict[str, Any]],
    format_weight: float = 0.4
) -> list[dict[str, float]]:
    
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch`.")

    scores = []
    c_list, I_list, format_list = [], [], []

    for reward_input in reward_inputs:
        response = reward_input["response"]
        gt = float(reward_input["ground_truth"])
        c_raw = extract_confidence(response)
        fmt_score = format_reward(response)

        c_list.append(c_raw if c_raw is not None else 0.0)
        I_list.append(gt)
        format_list.append(fmt_score)

    c_variance = statistics.variance(c_list) if len(c_list) > 1 else 0.0

    try:
        confidences_norm = [min(0.999, max(0.001, c / 10.0)) for c in c_list]
        auroc = roc_auc_score(I_list, confidences_norm)
    except ValueError:
        auroc = float("nan")

    ece = compute_ece(c_list, I_list)
    c_max = max(c_list)
    c_min = min(c_list)
    c_mean = statistics.mean(c_list)
    c_std = statistics.pstdev(c_list)

    correct_conf = [c_list[i] for i, acc in enumerate(I_list) if acc == 1]
    incorrect_conf = [c_list[i] for i, acc in enumerate(I_list) if acc == 0]
    if correct_conf and incorrect_conf:
        conf_gap = statistics.mean(correct_conf) - statistics.mean(incorrect_conf)
    else:
        conf_gap = float("nan")

    for idx in range(len(reward_inputs)):
        I = I_list[idx]
        c_val = c_list[idx]
        format_score_val = format_list[idx]

        reward_val = log_based_reward(c_val, bool(I))
        overall = reward_val

        scores.append({
            "overall": overall,
            "accuracy": I,
            "confidence_raw": c_val,
            "confidence_norm": min(0.999, max(0.001, c_val / 10.0)),
            "reward_val": reward_val,
            "format": format_score_val,
            "batch_conf_variance": c_variance,
            "batch_conf_auroc": auroc,
            "batch_conf_ece": ece,
            "batch_conf_max": c_max,
            "batch_conf_min": c_min,
            "batch_conf_mean": c_mean,
            "batch_conf_std": c_std,
            "batch_conf_gap_correct_incorrect": conf_gap
        })

    return scores
