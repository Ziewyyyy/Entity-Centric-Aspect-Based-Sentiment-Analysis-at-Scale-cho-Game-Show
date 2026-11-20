"""Logic thuần (không cần torch) cho model quan hệ HAS_OPINION.

- Dựng cặp ứng viên (entity, opinion) từ record đã export.
- Gom kết quả theo từng opinion, chấm điểm, chọn ngưỡng.
- Baseline "entity gần nhất" chính là hàm đang chạy trong serving/text_utils.py.
"""

import sys
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from serving import text_utils as tu  # noqa: E402

GroupKey = Tuple[str, int]  # (comment_id, opinion_index)


# ============================================================
# VALIDATE + BUILD PAIRS
# ============================================================

def validate_record(record):
    text = record["text"]

    for name in ("entities", "opinions"):
        previous_end = 0
        for span in record[name]:
            start, end = span["start"], span["end"]

            if not 0 <= start < end <= len(text):
                raise ValueError(f"Invalid {name} offsets: {span}")

            if text[start:end] != span["text"]:
                raise ValueError(f"{name} text mismatch: {span}")

            if start < previous_end:
                raise ValueError(f"Overlapping {name}: {span}")

            previous_end = end

    for relation in record["relations"]:
        if not 0 <= relation["entity"] < len(record["entities"]):
            raise ValueError(f"Bad entity index in relation: {relation}")

        if not 0 <= relation["opinion"] < len(record["opinions"]):
            raise ValueError(f"Bad opinion index in relation: {relation}")


def build_pairs(record) -> List[dict]:
    """Mọi cặp (opinion, entity) trong một comment; nhãn 1 nếu có HAS_OPINION."""
    gold = {(r["entity"], r["opinion"]) for r in record["relations"]}
    pairs = []

    for opinion_index, opinion in enumerate(record["opinions"]):
        for entity_index, entity in enumerate(record["entities"]):
            pairs.append({
                "comment_id": str(record["comment_id"]),
                "text": record["text"],
                "entity_index": entity_index,
                "opinion_index": opinion_index,
                "entity": entity,
                "opinion": opinion,
                "label": int((entity_index, opinion_index) in gold),
            })

    return pairs


# ============================================================
# GROUP BY OPINION
# ============================================================

def group_key(pair) -> GroupKey:
    return (pair["comment_id"], pair["opinion_index"])


def gold_groups(pairs: Sequence[dict]) -> Dict[GroupKey, Set[int]]:
    groups: Dict[GroupKey, Set[int]] = {}
    for pair in pairs:
        links = groups.setdefault(group_key(pair), set())
        if pair["label"] == 1:
            links.add(pair["entity_index"])
    return groups


def predicted_groups(
    pairs: Sequence[dict], probs: Sequence[float], threshold: float
) -> Dict[GroupKey, Set[int]]:
    groups: Dict[GroupKey, Set[int]] = {}
    for pair, prob in zip(pairs, probs):
        links = groups.setdefault(group_key(pair), set())
        if prob >= threshold:
            links.add(pair["entity_index"])
    return groups


def nearest_groups(records: Sequence[dict]) -> Dict[GroupKey, Set[int]]:
    """Baseline đang dùng: luôn chọn đúng 1 entity gần nhất (nếu có entity)."""
    groups: Dict[GroupKey, Set[int]] = {}

    for record in records:
        entities = record["entities"]
        if not entities:
            continue

        for opinion_index, opinion in enumerate(record["opinions"]):
            chosen = tu.link_opinion_to_entity(record["text"], opinion, entities)
            index = next(i for i, e in enumerate(entities) if e is chosen)
            groups[(str(record["comment_id"]), opinion_index)] = {index}

    return groups


def complete_keys(all_pairs: Sequence[dict], kept_pairs: Sequence[dict]) -> Set[GroupKey]:
    """Chỉ các opinion còn đủ mọi cặp (không cặp nào bị loại vì quá dài)."""
    total: Dict[GroupKey, int] = {}
    kept: Dict[GroupKey, int] = {}

    for pair in all_pairs:
        total[group_key(pair)] = total.get(group_key(pair), 0) + 1
    for pair in kept_pairs:
        kept[group_key(pair)] = kept.get(group_key(pair), 0) + 1

    return {key for key, n in total.items() if kept.get(key, 0) == n}


def multi_entity_keys(pairs: Sequence[dict]) -> Set[GroupKey]:
    """Opinion nằm trong comment có từ 2 entity trở lên (ca khó)."""
    entity_count: Dict[str, Set[int]] = {}
    for pair in pairs:
        entity_count.setdefault(pair["comment_id"], set()).add(pair["entity_index"])

    return {
        group_key(pair)
        for pair in pairs
        if len(entity_count[pair["comment_id"]]) >= 2
    }


# ============================================================
# SCORING
# ============================================================

def score_groups(
    gold: Dict[GroupKey, Set[int]],
    predicted: Dict[GroupKey, Set[int]],
    keys: Set[GroupKey],
) -> dict:
    keys = {k for k in keys if k in gold and k in predicted}

    exact = sum(gold[k] == predicted[k] for k in keys)

    tp = sum(len(gold[k] & predicted[k]) for k in keys)
    fp = sum(len(predicted[k] - gold[k]) for k in keys)
    fn = sum(len(gold[k] - predicted[k]) for k in keys)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "opinions": len(keys),
        "exact_match": exact / len(keys) if keys else 0.0,
        "link_precision": precision,
        "link_recall": recall,
        "link_f1": f1,
    }


def find_best_threshold(
    pairs: Sequence[dict],
    probs: Sequence[float],
    keys: Set[GroupKey],
    grid: Sequence[float] = tuple(round(0.30 + 0.05 * i, 2) for i in range(11)),
) -> Tuple[float, List[dict]]:
    """Chọn ngưỡng theo link F1 trên validation; hòa thì chọn ngưỡng gần 0.5."""
    gold = gold_groups(pairs)
    table = []

    for threshold in grid:
        predicted = predicted_groups(pairs, probs, threshold)
        score = score_groups(gold, predicted, keys)
        table.append({"threshold": threshold, **score})

    best = max(table, key=lambda row: (row["link_f1"], -abs(row["threshold"] - 0.5)))
    return best["threshold"], table
