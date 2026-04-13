import re
import statistics
from typing import Any

from mathruler.grader import extract_boxed_content, grade_answer


def extract_confidence(response: str) -> float:
    
    match = re.search(r"<confidence>(\d+(?:\.\d+)?)</confidence>", response)
    if match:
        c = float(match.group(1))
        return max(0.0, min(c / 10.0, 1.0))
    return 0.5


def format_reward(response: str) -> float:
    
    pattern = re.compile(
        r"<think>.*?</think>.*?\\boxed\{.*?\}.*?<think>.*?</think>.*?<confidence>.*?</confidence>",
        re.DOTALL,
    )
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_score(
    reward_inputs: list[dict[str, Any]],
    brier_weight: float = 0.8,
    format_weight: float = 0.4,
) -> list[dict[str, float]]:
    
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    scores = []

    c_list = []
    processed_responses = []
    I_list = []
    format_list = []
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

    for idx, (response, gt) in enumerate(processed_responses):
        I = I_list[idx]
        c = c_list[idx]
        format_score = format_list[idx]
        brier_score = -((c - I) ** 2)
        overall = I + brier_weight * brier_score + format_weight * format_score

        scores.append(
            {
                "overall": overall,
                "accuracy": I,
                "brier": brier_score,
                "confidence": c,
                "format": format_score,
                "batch_conf_variance": c_variance,
            }
        )

    return scores
