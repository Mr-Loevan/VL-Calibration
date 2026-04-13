import re
from typing import Any
from mathruler.grader import extract_boxed_content, grade_answer
import statistics


def extract_confidence(response: str) -> float:
    
    match = re.search(r"<confidence>(\d+(?:\.\d+)?)</confidence>", response)
    if match:
        c = float(match.group(1)) 
        return max(0.0, min((c / 10.0) ** 1.3, 1.0))
    return 0.5

def format_reward(response: str) -> float:
    
    pattern = re.compile(
        r"<think>.*?</think>.*?\\boxed\{.*?\}.*?<analysis>.*?</analysis>.*?<confidence>(.*?)</confidence>",
        re.DOTALL,
    )
    match = re.fullmatch(pattern, response)

    if not match:
        return 0.0

    try:
        conf_str = match.group(1).strip()
        conf_value = float(conf_str)
        if 1.0 <= conf_value <= 10.0:
            return 1.0
        else:
            return 0.5
    except ValueError:
        return 0.0
    


def accuracy_reward(response: str, ground_truth: str) -> float:
    
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


import re
import math
import statistics
from typing import Any
from sklearn.metrics import roc_auc_score
import numpy as np

def compute_score(
    reward_inputs: list[dict[str, Any]],
    brier_weight: float = 0.8,
    format_weight: float = 0.4,
) -> list[dict[str, float]]:
    
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    scores = []
    c_list, I_list, format_list, processed_responses = [], [], [], []

    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
        gt = reward_input["ground_truth"]
        processed_responses.append((response, gt))
        c_list.append(extract_confidence(response))
        I_list.append(accuracy_reward(response, gt))
        format_list.append(format_reward(response))

    if len(c_list) > 1:
        c_variance = statistics.variance(c_list)
    else:
        c_variance = 0.0

    try:
        auroc = roc_auc_score(I_list, c_list)
    except ValueError:
        auroc = float("nan")

    def compute_ece(confidences, accuracies, n_bins=10):
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            start, end = bins[i], bins[i + 1]
            idx = [j for j, c in enumerate(confidences) if start <= c < end]
            if not idx:
                continue
            avg_conf = np.mean([confidences[j] for j in idx])
            avg_acc = np.mean([accuracies[j] for j in idx])
            ece += abs(avg_conf - avg_acc) * len(idx) / len(confidences)
        return ece

    ece = compute_ece(c_list, I_list)

    c_max = max(c_list)
    c_min = min(c_list)
    c_mean = statistics.mean(c_list)
    c_std = statistics.pstdev(c_list)
    c_entropy = -sum(c * math.log(c + 1e-8) + (1 - c) * math.log(1 - c + 1e-8) for c in c_list) / len(c_list)

    correct_conf = [c_list[i] for i, acc in enumerate(I_list) if acc == 1]
    incorrect_conf = [c_list[i] for i, acc in enumerate(I_list) if acc == 0]

    if correct_conf and incorrect_conf:
        conf_gap = statistics.mean(correct_conf) - statistics.mean(incorrect_conf)
    else:
        conf_gap = float("nan")

    for idx, (response, gt) in enumerate(processed_responses):
        I = I_list[idx]
        c = c_list[idx]
        format_score = format_list[idx]
        brier_score = -(c - I) ** 2
        overall = I + brier_weight * brier_score + format_weight * format_score

        scores.append(
            {
                "overall": overall,
                "accuracy": I,
                "brier": brier_score,
                "confidence": c,
                "format": format_score,
                "batch_conf_variance": c_variance,
                "batch_conf_auroc": auroc,
                "batch_conf_ece": ece,
                "batch_conf_max": c_max,
                "batch_conf_min": c_min,
                "batch_conf_mean": c_mean,
                "batch_conf_std": c_std,
                "batch_conf_gap_correct_incorrect": conf_gap,
            }
        )

    return scores



