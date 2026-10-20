import re

from transformers import AutoTokenizer

MODEL_NAME = "vinai/phobert-base-v2"
MAX_LENGTH = 256

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=False
)


def segment_with_offsets(text):
    words = []

    for match in re.finditer(r"\w+|[^\w\s]", text, re.UNICODE):
        words.append({
            "text": match.group(),
            "start": match.start(),
            "end": match.end()
        })

    return words


def encode_words(text):
    words = segment_with_offsets(text)

    input_ids = [tokenizer.cls_token_id]
    word_positions = []

    for word in words:
        token_text = word["text"]

        prefix = " " if len(input_ids) > 1 else ""

        subword_ids = tokenizer.encode(
            prefix + token_text,
            add_special_tokens=False
        )

        if not subword_ids:
            raise ValueError(
                f"Cannot tokenize word: {word!r}"
            )

        word_positions.append(len(input_ids))
        input_ids.extend(subword_ids)

    input_ids.append(tokenizer.sep_token_id)

    if len(input_ids) > MAX_LENGTH:
        raise ValueError(
            f"Sequence length {len(input_ids)} exceeds "
            f"MAX_LENGTH={MAX_LENGTH}"
        )

    return {
        "words": words,
        "input_ids": input_ids,
        "word_positions": word_positions
    }