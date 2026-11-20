
"""Text utilities for entity-opinion relation classification."""

from typing import Dict, List, Sequence


# Special markers used by PhoBERT.
RELATION_MARKERS = [
    "<ent>",
    "</ent>",
    "<op>",
    "</op>",
]


def build_relation_input(
    text: str,
    entity: Dict,
    opinion: Dict,
) -> str:
    """
    Đánh dấu Entity và Opinion trong comment.

    Format:
        <ent> Entity </ent> ... <op> Opinion </op>
        </s> Entity: TYPE
    """

    entity_start = entity["start"]
    entity_end = entity["end"]

    opinion_start = opinion["start"]
    opinion_end = opinion["end"]

    # Validate offsets.
    for name, start, end in (
        ("entity", entity_start, entity_end),
        ("opinion", opinion_start, opinion_end),
    ):
        if not 0 <= start < end <= len(text):
            raise ValueError(f"Invalid {name} offsets: {start}-{end}")

    # Validate span text.
    if text[entity_start:entity_end] != entity["text"]:
        raise ValueError("Entity text does not match its offsets")

    if text[opinion_start:opinion_end] != opinion["text"]:
        raise ValueError("Opinion text does not match its offsets")

    # Entity and opinion spans must not overlap.
    if max(entity_start, opinion_start) < min(entity_end, opinion_end):
        raise ValueError("Entity and opinion spans overlap")

    # Insert markers from right to left to preserve original offsets.
    insertions = [
        (entity_start, "<ent> "),
        (entity_end, " </ent>"),
        (opinion_start, "<op> "),
        (opinion_end, " </op>"),
    ]

    marked_text = text

    for position, marker in sorted(
        insertions,
        key=lambda item: item[0],
        reverse=True,
    ):
        marked_text = (
            marked_text[:position]
            + marker
            + marked_text[position:]
        )

    entity_type = entity.get("type", "UNKNOWN")

    return f"{marked_text} </s> Entity: {entity_type}"


def link_opinion_to_entity(
    text: str,
    opinion: Dict,
    entities: Sequence[Dict],
) -> Dict:
    """
    Baseline: chọn Entity có khoảng cách ký tự gần Opinion nhất.

    Nếu hai Entity có cùng khoảng cách, chọn Entity xuất hiện
    trước trong danh sách.
    """

    if not entities:
        raise ValueError("Cannot select an entity from an empty list")

    opinion_start = opinion["start"]
    opinion_end = opinion["end"]

    def distance(entity: Dict) -> int:
        entity_start = entity["start"]
        entity_end = entity["end"]

        if entity_end <= opinion_start:
            return opinion_start - entity_end

        if opinion_end <= entity_start:
            return entity_start - opinion_end

        # Overlapping spans have distance zero.
        return 0

    return min(entities, key=distance)
