
import json
from pathlib import Path
from collections import Counter

INPUT_PATH = Path("data/labeled/label_studio_export.json")
OUTPUT_PATH = Path("data/labeled/opinion_spans.jsonl")

OPINION_LABEL = "OPINION"


# ============================================================
# NORMALIZE EXPORTED SPAN TEXT
# ============================================================

def normalize_span(text):
    """
    Chuẩn hóa whitespace khi so sánh Opinion Span.

    Xử lý:
    - Khoảng trắng đầu/cuối
    - Xuống dòng thực: \n
    - Chuỗi escape: \\n
    - Tab: \t hoặc \\t
    - Carriage return: \r hoặc \\r

    Không thay đổi nội dung ở giữa câu.
    """
    if not isinstance(text, str):
        return text

    return (
        text.replace("\\r\\n", "\n")
            .replace("\\n", "\n")
            .replace("\\r", "\r")
            .replace("\\t", "\t")
            .strip()
    )


# ============================================================
# EXTRACT OPINIONS
# ============================================================

def extract_opinions(task):
    task_id = task.get("id")
    data = task.get("data", {})
    text = data.get("text", "")

    annotations = task.get("annotations", [])

    if not isinstance(text, str):
        raise ValueError(
            f"Task {task_id}: invalid text"
        )

    # Chỉ lấy annotation đã hoàn thành.
    valid_annotations = [
        annotation
        for annotation in annotations
        if not annotation.get("was_cancelled", False)
        and annotation.get("result") is not None
    ]

    if not valid_annotations:
        return None

    if len(valid_annotations) > 1:
        raise ValueError(
            f"Task {task_id} has multiple annotations. "
            "Choose an annotation policy before exporting."
        )

    annotation = valid_annotations[0]
    opinions = []

    for result in annotation.get("result", []):

        if result.get("type") != "labels":
            continue

        value = result.get("value", {})
        labels = value.get("labels", [])

        if OPINION_LABEL not in labels:
            continue

        start = value.get("start")
        end = value.get("end")

        # ====================================================
        # VALIDATE OFFSETS
        # ====================================================

        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or not 0 <= start < end <= len(text)
        ):
            raise ValueError(
                f"Invalid opinion offsets in task {task_id}: "
                f"{value}"
            )

        span_text = text[start:end]

        # ====================================================
        # CHECK EXPORTED TEXT
        # ====================================================

        exported_text = value.get("text")

        if exported_text is not None:

            if not isinstance(exported_text, str):
                raise ValueError(
                    f"Invalid exported opinion text "
                    f"in task {task_id}"
                )

            if exported_text != span_text:

                normalized_exported = normalize_span(
                    exported_text
                )

                normalized_actual = normalize_span(
                    span_text
                )

                if normalized_exported == normalized_actual:

                    print(
                        f"[WARNING] Task {task_id}: "
                        "Opinion text differs only by "
                        "whitespace or escaped whitespace. "
                        "Using original text offsets."
                    )

                else:
                    raise ValueError(
                        f"Offset mismatch in task {task_id}: "
                        f"exported={exported_text!r}, "
                        f"actual={span_text!r}"
                    )

        # ====================================================
        # TRIM WHITESPACE AND UPDATE OFFSETS
        # ====================================================

        left_trim = (
            len(span_text)
            - len(span_text.lstrip())
        )

        right_trim = (
            len(span_text)
            - len(span_text.rstrip())
        )

        normalized_start = start + left_trim
        normalized_end = end - right_trim

        if normalized_start >= normalized_end:
            raise ValueError(
                f"Empty opinion after trimming "
                f"in task {task_id}"
            )

        normalized_text = text[
            normalized_start:normalized_end
        ]

        # Kiểm tra lại để bảo đảm offset chính xác.
        if text[normalized_start:normalized_end] != normalized_text:
            raise ValueError(
                f"Final opinion offset mismatch "
                f"in task {task_id}"
            )

        opinions.append({
            "text": normalized_text,
            "start": normalized_start,
            "end": normalized_end
        })

    # ========================================================
    # SORT OPINIONS
    # ========================================================

    opinions.sort(
        key=lambda opinion: (
            opinion["start"],
            opinion["end"]
        )
    )

    # ========================================================
    # CHECK OVERLAPPING OPINIONS
    # ========================================================

    for previous, current in zip(
        opinions,
        opinions[1:]
    ):

        if current["start"] < previous["end"]:
            raise ValueError(
                f"Overlapping opinions in task {task_id}: "
                f"{previous} and {current}"
            )

    # ========================================================
    # RETURN RECORD
    # ========================================================

    return {
        "comment_id": data.get(
            "comment_id",
            str(task_id)
        ),
        "video_id": data.get("video_id"),
        "video_type": data.get("video_type"),
        "text": text,
        "opinions": opinions
    }


# ============================================================
# MAIN
# ============================================================

def main():

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"File not found: {INPUT_PATH}"
        )

    tasks = json.loads(
        INPUT_PATH.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(tasks, list):
        raise ValueError(
            "Expected a JSON array of Label Studio tasks"
        )

    records = []
    skipped = 0

    for task in tasks:

        record = extract_opinions(task)

        if record is None:
            skipped += 1
            continue

        records.append(record)

    # ========================================================
    # SAVE JSONL
    # ========================================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8"
    ) as f:

        for record in records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )

    # ========================================================
    # STATISTICS
    # ========================================================

    counts = Counter()

    for record in records:

        counts["opinions"] += len(
            record["opinions"]
        )

        if not record["opinions"]:
            counts["no_opinion"] += 1

    # ========================================================
    # PRINT REPORT
    # ========================================================

    print("\n" + "=" * 60)
    print("OPINION EXPORT")
    print("=" * 60)

    print("Tasks:", len(tasks))
    print("Exported comments:", len(records))
    print("Skipped (no annotation):", skipped)
    print("Opinion spans:", counts["opinions"])
    print(
        "Comments without opinion:",
        counts["no_opinion"]
    )
    print("Saved to:", OUTPUT_PATH)

    print("=" * 60)


if __name__ == "__main__":
    main()
