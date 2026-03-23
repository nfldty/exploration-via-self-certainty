import re
import math


def _normalize_answer(answer: str) -> str:
    """Normalize a mathematical answer string for comparison."""
    if answer is None:
        return None
    answer = answer.strip()
    answer = answer.replace(" ", "")
    answer = answer.replace("\\left", "").replace("\\right", "")
    answer = answer.replace("\\!", "").replace("\\,", "").replace("\\;", "")
    answer = answer.replace("\\text{", "").replace("\\mathrm{", "").replace("\\textbf{", "")
    answer = answer.replace("\\dfrac", "\\frac")
    answer = answer.replace("\\tfrac", "\\frac")
    answer = answer.replace("\\%", "")
    answer = answer.replace("$", "")
    if answer.endswith("."):
        answer = answer[:-1]
    if answer.endswith("}"):
        open_braces = answer.count("{")
        close_braces = answer.count("}")
        if close_braces > open_braces:
            answer = answer[:-(close_braces - open_braces)]
    return answer


def _try_parse_number(s: str) -> float | None:
    """Try to parse a string as a number, handling fractions."""
    s = s.strip().replace(",", "")
    try:
        return float(s)
    except ValueError:
        pass
    frac_match = re.match(r'^\\frac\{([^}]+)\}\{([^}]+)\}$', s)
    if frac_match:
        try:
            num = float(frac_match.group(1))
            den = float(frac_match.group(2))
            if den != 0:
                return num / den
        except ValueError:
            pass
    neg_frac = re.match(r'^-\\frac\{([^}]+)\}\{([^}]+)\}$', s)
    if neg_frac:
        try:
            num = float(neg_frac.group(1))
            den = float(neg_frac.group(2))
            if den != 0:
                return -num / den
        except ValueError:
            pass
    return None


def extract_solution(solution_str: str) -> str | None:
    """Extract the answer from a solution string, looking for \\boxed{...}."""
    if solution_str is None:
        return None
    boxed_matches = list(re.finditer(r'\\boxed\{', solution_str))
    if not boxed_matches:
        return None
    last_match = boxed_matches[-1]
    start = last_match.end()
    depth = 1
    i = start
    while i < len(solution_str) and depth > 0:
        if solution_str[i] == '{':
            depth += 1
        elif solution_str[i] == '}':
            depth -= 1
        i += 1
    if depth == 0:
        return solution_str[start:i - 1]
    return None


def _answers_match(pred: str, gt: str) -> bool:
    """Check if predicted and ground truth answers match."""
    norm_pred = _normalize_answer(pred)
    norm_gt = _normalize_answer(gt)
    if norm_pred is None or norm_gt is None:
        return False
    if norm_pred == norm_gt:
        return True
    num_pred = _try_parse_number(norm_pred)
    num_gt = _try_parse_number(norm_gt)
    if num_pred is not None and num_gt is not None:
        return math.isclose(num_pred, num_gt, rel_tol=1e-6, abs_tol=1e-8)
    return False


def compute_score(solution_str, ground_truth, method='strict', error_score=0.0, format_score=0.1, score=1.0):
    """Scoring function for MATH dataset.

    Args:
        solution_str: the full solution text from the model (prompt + response)
        ground_truth: the ground truth answer string (extracted from \\boxed{} in the dataset)
        method: extraction method (unused, kept for interface compatibility)
        error_score: score when no answer is found
        format_score: score when answer is found but incorrect
        score: score when answer is correct
    """
    # Only search the assistant's response, not the prompt (which contains \boxed{}).
    assistant_marker = '<|im_start|>assistant'
    marker_pos = solution_str.rfind(assistant_marker)
    if marker_pos != -1:
        response_str = solution_str[marker_pos + len(assistant_marker):]
    else:
        response_str = solution_str

    pred_answer = extract_solution(response_str)
    if pred_answer is None:
        return error_score
    if _answers_match(pred_answer, ground_truth):
        return score
    return format_score
