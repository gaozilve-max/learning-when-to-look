import os
import sys
import json
import math
import re
import importlib.util  # ← 必须有这一行
from typing import Any, Dict, List

# ==== 工具函数（math_verify.py 动态导入） ====
_current_file_dir = os.path.dirname(os.path.abspath(__file__))
_math_verify_path = os.path.join(_current_file_dir, "math_verify.py")

if os.path.exists(_math_verify_path):
    _math_verify_spec = importlib.util.spec_from_file_location("math_verify_module", _math_verify_path)
    _math_verify_module = importlib.util.module_from_spec(_math_verify_spec)
    sys.modules["math_verify_module"] = _math_verify_module
    _math_verify_spec.loader.exec_module(_math_verify_module)
    _format_reward = _math_verify_module.format_reward
    _accuracy_reward = _math_verify_module.accuracy_reward
else:
    raise ImportError(f"Cannot find math_verify.py at {_math_verify_path}")

# ==== 关键奖励相关函数 ====

def count_words(text: str) -> int:
    if not text or not text.strip():
        return 0
    words = re.findall(r'\b\w+\b', text)
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
    return len(words) + len(chinese_chars)

def extract_perceptions(text: str) -> list[str]:
    pattern = r"<perception>(.*?)</perception>"
    matches = re.findall(pattern, text or "", re.DOTALL)
    return [m.strip() for m in matches if m and m.strip()]

def format_reward_with_perception(response: str) -> float:
    think_ok = "<think>" in response and "</think>" in response
    if not think_ok:
        return 0.0
    open_count = response.count("<perception>")
    close_count = response.count("</perception>")
    if open_count != close_count:
        return 0.0
    return 1.0

def length_reward_l1_max(
    n_y: int,
    n_gold: int = 2048,
    alpha: float = 0.005,
    delta: float = 0.5,
) -> float:
    if n_y <= n_gold:
        return 1.0
    raw = delta - alpha * (n_y - n_gold)
    return max(0.0, min(1.0, raw))

def perception_count_reward_by_words(
    word_count: int,
    actual_perception_count: int,
    words_per_perception: int = 120,
    is_correct: bool = False,
    penalty_per_excess: float = 0.2,
) -> float:
    if not is_correct:
        return 0.0
    if word_count == 0:
        return 0.0
    target_perception_count = math.ceil(word_count / words_per_perception)
    if target_perception_count == 0:
        return 1.0 if actual_perception_count == 0 else 0.0
    if actual_perception_count <= target_perception_count:
        score = actual_perception_count / target_perception_count
        return min(1.0, max(0.0, score))
    excess_count = actual_perception_count - target_perception_count
    base_score = 1.0
    penalty = excess_count * penalty_per_excess
    score = base_score - penalty
    return max(0.0, min(1.0, score))

def _get_thinking_keywords_patterns() -> list[str]:
    """扩展的关键词模式，包括思考相关、步骤相关和特殊token相关"""
    phrases = [
        # 原有思考相关关键词
        r"let me recalculate", r"let'?s revise", r"let me reconsider", r"let'?s rethink", r"let me check again",
        r"double-?check", r"verify", r"re-?evaluate", r"reassess", r"revisit", r"let'?s try again",
        r"\bwait\b", r"hold on", r"\bhowever\b", r"\bbut\b", r"\balthough\b", r"on second thought", r"wait a second",
        r"is that right", r"did I miss something", r"I'?m not sure", r"I am not sure", r"I wonder", r"perhaps", r"maybe",
        r"\bI think\b", r"\bmistake\b", r"\berror\b", r"\bincorrect\b", r"\bwrong\b", r"I was mistaken", r"that'?s not right",
        r"this contradicts", r"I need to reconsider", r"doesn'?t make sense", r"I made an error", r"\bcontradiction\b",
        r"\bflaw\b", r"\binvalid\b",
        
        # 步骤相关关键词
        r"\bfirst\b", r"\bsecond\b", r"\bthird\b", r"\bfourth\b", r"\bfifth\b", r"\bsixth\b", r"\bseventh\b", r"\beighth\b",
        r"\bninth\b", r"\btenth\b", r"\bstep\b", r"\bsteps\b", r"\bstep\s+\d+", r"\bstep\s+[a-z]",
        r"\boption\b", r"\boptions\b", r"\bapproach\b", r"\bapproaches\b", r"\bmethod\b", r"\bmethods\b",
        r"\bway\b", r"\bways\b", r"\bprocess\b", r"\bprocedure\b", r"\bstage\b", r"\bstages\b", r"\bphase\b", r"\bphases\b",
        r"\bpart\b", r"\bparts\b", r"\bsection\b", r"\bsections\b", r"\baspect\b", r"\baspects\b",
        r"\bnext\b", r"\bthen\b", r"\bafter\b", r"\bfollowing\b", r"\bsubsequently\b",
        r"\binitially\b", r"\bfinally\b", r"\blastly\b", r"\bultimately\b",
        r"\bsequence\b", r"\border\b", r"\bsequence\b", r"\bprogression\b",
        
        # 特殊token相关
        r"<think>",

        
        # 观察和分析相关关键词
        r"\bobserve\b", r"\bobserving\b", r"\bobservation\b", r"\bobservations\b",
        r"\bnotice\b", r"\bnoticing\b", r"\bnoticed\b",
        r"\bsee\b", r"\bseeing\b", r"\bsaw\b", r"\bseen\b",
        r"\blook\b", r"\blooking\b", r"\blooked\b",
        r"\bexamine\b", r"\bexamining\b", r"\bexamined\b", r"\bexamination\b",
        r"\bcheck\b", r"\bchecking\b", r"\bchecked\b",
        r"\bidentify\b", r"\bidentifying\b", r"\bidentified\b", r"\bidentification\b",
        r"\brecognize\b", r"\brecognizing\b", r"\brecognized\b", r"\brecognition\b",
        r"\bdetect\b", r"\bdetecting\b", r"\bdetected\b", r"\bdetection\b",
        r"\bfind\b", r"\bfinding\b", r"\bfound\b",
        r"\bdiscover\b", r"\bdiscovering\b", r"\bdiscovered\b", r"\bdiscovery\b",
        r"\banalyze\b", r"\banalyzing\b", r"\banalyzed\b", r"\banalysis\b",
        r"\bconsider\b", r"\bconsidering\b", r"\bconsidered\b", r"\bconsideration\b",
        r"\bevaluate\b", r"\bevaluating\b", r"\bevaluated\b", r"\bevaluation\b",
        r"\bassess\b", r"\bassessing\b", r"\bassessed\b", r"\bassessment\b",
        r"\binspect\b", r"\binspecting\b", r"\binspected\b", r"\binspection\b",
        r"\breview\b", r"\breviewing\b", r"\breviewed\b",
        r"\bstudy\b", r"\bstudying\b", r"\bstudied\b",
        r"\binvestigate\b", r"\binvestigating\b", r"\binvestigated\b", r"\binvestigation\b",
        r"\bexplore\b", r"\bexploring\b", r"\bexplored\b", r"\bexploration\b",
        r"\bexamine\b", r"\bexamining\b", r"\bexamined\b",
        
        # 中文思考相关关键词
        r"重新思考", r"重新考虑", r"重新计算", r"重新审视", r"重新评估", r"重新检查", r"再次思考", r"再次考虑",
        r"让我重新", r"让我再想想", r"让我再思考", r"让我重新考虑", r"让我重新计算", r"重新验证", r"再想想", r"再考虑", r"等等", r"等一下",
        r"不过", r"但是", r"然而", r"虽然", r"或许", r"可能", r"也许", r"我想", r"我觉得", r"我不确定", r"我不太确定", r"我怀疑", r"我疑惑",
        r"我困惑", r"这不对", r"这好像不对", r"这可能不对", r"这是对的吗", r"对吗", r"是不是", r"会不会", r"错误", r"不对", r"不正确", r"有问题",
        r"有误", r"失误", r"我错了", r"我弄错了", r"我理解错了", r"我想错了", r"这不合理", r"这说不通", r"这不符合", r"矛盾", r"有矛盾",
        r"存在矛盾", r"有缺陷", r"无效", r"不成立", r"需要重新", r"需要再想想", r"需要重新思考", r"让我看看", r"让我想想", r"让我思考", r"仔细想想",
        r"仔细思考", r"仔细考虑", r"再仔细看看", r"再仔细想想", r"回头看看", r"回过头看", r"另一方面", r"换个角度", r"换个思路", r"换个方式",
        r"或者", r"或者说", r"也就是说", r"换句话说",
        
        # 中文步骤相关关键词
        r"第一", r"第二", r"第三", r"第四", r"第五", r"第六", r"第七", r"第八", r"第九", r"第十",
        r"第一步", r"第二步", r"第三步", r"步骤", r"阶段", r"阶段一", r"阶段二", r"阶段三",
        r"方法", r"方式", r"途径", r"途径一", r"途径二", r"选项", r"选项一", r"选项二",
        r"部分", r"部分一", r"部分二", r"方面", r"方面一", r"方面二",
        r"接下来", r"然后", r"之后", r"随后", r"接着",
        r"首先", r"最后", r"最终", r"最后一步",
        r"顺序", r"次序", r"流程", r"过程",
        
        # 中文观察和分析相关关键词
        r"观察", r"注意到", r"看到", r"看到", r"查看", r"检查", r"识别", r"辨认", r"检测", r"发现",
        r"分析", r"考虑", r"评估", r"评定", r"审视", r"审查", r"研究", r"调查", r"探索", r"检验",
    ]
    return phrases

def _split_sentences(text: str) -> list[str]:
    if not text:
        return []
    t = re.sub(r"\s+", " ", text.strip())
    parts = re.split(r"(?<=[\.!?。？！])\s+", t)
    return [p.strip() for p in parts if p and p.strip()]

def _extract_perception_starts(response: str) -> list[int]:
    return [m.start() for m in re.finditer(r"<perception>", response or "")]

def perception_rethink_score(response: str) -> float:
    """计算包含反思关键词的perception比例"""
    if not response:
        return 0.0
    starts = _extract_perception_starts(response)
    total = len(starts)
    if total == 0:
        return 0.0
    patterns = [re.compile(p, re.IGNORECASE) for p in _get_thinking_keywords_patterns()]
    hits = 0
    for s in starts:
        prefix = response[:s]
        sentences = _split_sentences(prefix)
        if not sentences:
            continue
        prev_sentence = sentences[-1]
        if any(p.search(prev_sentence) for p in patterns):
            hits += 1
    return hits / total if total > 0 else 0.0

def count_reflect_perceptions(response: str) -> int:
    """统计包含反思关键词的perception数量"""
    if not response:
        return 0
    starts = _extract_perception_starts(response)
    if len(starts) == 0:
        return 0
    patterns = [re.compile(p, re.IGNORECASE) for p in _get_thinking_keywords_patterns()]
    reflect_count = 0
    for s in starts:
        prefix = response[:s]
        sentences = _split_sentences(prefix)
        if not sentences:
            continue
        prev_sentence = sentences[-1]
        if any(p.search(prev_sentence) for p in patterns):
            reflect_count += 1
    return reflect_count

def apply_reflect_penalty_to_rethink_score(
    rethink_score: float,
    reflect_count: int,
    total_perception_count: int,
    max_allowed_ratio: float = 0.5,
    penalty_per_excess: float = 0.2,
) -> float:
    """
    当reflect perception比例达到阈值时直接给满分，否则保持原始rethink score
    
    Args:
        rethink_score: 原始的rethink分数
        reflect_count: 包含反思关键词的perception数量
        total_perception_count: 总perception数量
        max_allowed_ratio: 达到满分所需的reflect比例（默认0.5，即一半）
        penalty_per_excess: 保留该参数以兼容旧接口（不再使用）
    
    Returns:
        调整后的rethink分数
    """
    if total_perception_count == 0:
        return rethink_score
    reflect_ratio = reflect_count / total_perception_count
    if reflect_ratio >= max_allowed_ratio:
        return 1.0
    return rethink_score

# ==== 主奖励函数 ====

def compute_score(
    reward_inputs: list[dict[str, Any]],
    format_weight: float = 0.05,
    perception_count_weight: float = 0.05,
    perception_rethink_weight: float = 0.05,
    length_weight: float = 0.05,
    images_key: str = "images",
    n_gold: int = 2048,
    words_per_perception: int = 120,
    max_reflect_ratio: float = 0.5,
    reflect_penalty_per_excess: float = 0.2,
) -> list[dict[str, float]]:
    """
    组合奖励函数（v2版本，perception分为两个独立的奖励项）：
      overall = accuracy_weight * accuracy
                + format_weight * format
                + perception_count_weight * perception_count_score
                + perception_rethink_weight * perception_rethink
                + length_weight * length

    - perception_count_score: 基于词数计算perception数量奖励
    - perception_rethink_score: 统计每个 <perception> 前一句是否包含反思关键词，命中比例
      当reflect perception比例达到阈值（默认0.5）时，直接给满分1.0

    其中 accuracy_weight = 1 - format_weight - perception_count_weight - perception_rethink_weight - length_weight
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for perception_rethink reward function.")
    
    format_weight = float(format_weight)
    perception_count_weight = float(perception_count_weight)
    perception_rethink_weight = float(perception_rethink_weight)
    length_weight = float(length_weight)
    n_gold = int(n_gold)
    words_per_perception = int(words_per_perception)
    max_reflect_ratio = float(max_reflect_ratio)
    reflect_penalty_per_excess = float(reflect_penalty_per_excess)

    accuracy_weight = 1.0 - format_weight - perception_count_weight - perception_rethink_weight - length_weight
    if accuracy_weight < 0:
        raise ValueError(
            f"Invalid weights: format_weight={format_weight}, perception_count_weight={perception_count_weight}, "
            f"perception_rethink_weight={perception_rethink_weight}, length_weight={length_weight}, "
            f"sum={format_weight + perception_count_weight + perception_rethink_weight + length_weight} > 1"
        )

    batch_size = len(reward_inputs)
    scores: list[dict[str, float]] = []

    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
        format_score = float(format_reward_with_perception(response))
        accuracy_score = float(_accuracy_reward(response, reward_input["ground_truth"]))
        response_length = int(reward_input.get("response_length", 0))
        length_score = float(length_reward_l1_max(response_length, n_gold))
        if accuracy_score != 1.0:
            length_score = 0.0

        word_count = count_words(response)
        perceptions = extract_perceptions(response)
        perception_count = len(perceptions)
        perception_count_score = float(
            perception_count_reward_by_words(
                word_count=word_count,
                actual_perception_count=perception_count,
                words_per_perception=words_per_perception,
                is_correct=(accuracy_score == 1.0),
            )
        )

        # 计算perception_rethink_score
        perception_rethink_raw = float(perception_rethink_score(response))
        
        # 计算reflect perception数量并应用调整到rethink_score
        reflect_count = count_reflect_perceptions(response)
        perception_rethink = apply_reflect_penalty_to_rethink_score(
            rethink_score=perception_rethink_raw,
            reflect_count=reflect_count,
            total_perception_count=perception_count,
            max_allowed_ratio=max_reflect_ratio,
            penalty_per_excess=reflect_penalty_per_excess,
        )

        # 计算总分（perception_count和perception_rethink作为两个独立的奖励项）
        overall_score = (
            accuracy_weight * accuracy_score
            + format_weight * format_score
            + perception_count_weight * perception_count_score
            + perception_rethink_weight * perception_rethink
            + length_weight * length_score
        )
        # 确保总分在[0, 1]范围内
        overall_score = max(0.0, min(1.0, overall_score))

        scores.append(
            {
                "overall": float(overall_score),
                "format": float(format_score),
                "accuracy": float(accuracy_score),
                "perception_count": float(perception_count_score),
                "perception_rethink": float(perception_rethink),
                "perception_rethink_raw": float(perception_rethink_raw),
                "perception_count_raw": float(perception_count),
                "reflect_count": float(reflect_count),
                "word_count": float(word_count),
                "target_perception_count": float(math.ceil(word_count / words_per_perception) if word_count > 0 else 0),
                "length": float(length_score),
            }
        )
    return scores

