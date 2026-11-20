
import argparse
import json
import re
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
)

MODEL_PATH = Path("D:/AI_Models/opinion_phobert")
MAX_LENGTH = 256

# Phải giống cách tách token trong train_opinion.py
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)

ID2LABEL = {
    0: "O",
    1: "B-OPINION",
    2: "I-OPINION",
}


def tokenize_with_offsets(text):
    return [
        {
            "text": match.group(),
            "start": match.start(),
            "end": match.end(),
        }
        for match in TOKEN_PATTERN.finditer(text)
    ]


class OpinionPredictor:
    def __init__(self, model_path=MODEL_PATH):
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"Loading model from: {model_path}")
        print(f"Device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            use_fast=False,
        )

        self.model = (
            AutoModelForTokenClassification
            .from_pretrained(str(model_path))
        )

        self.model.to(self.device)
        self.model.eval()

        self.id2label = {
            int(k): v
            for k, v in self.model.config.id2label.items()
        }

        print("Model loaded successfully!")

    def encode_text(self, text):
        words = tokenize_with_offsets(text)

        input_ids = [self.tokenizer.cls_token_id]

        # Lưu vị trí subword đầu tiên của mỗi word
        word_positions = []

        for word in words:
            token_ids = self.tokenizer.encode(
                word["text"],
                add_special_tokens=False,
            )

            if not token_ids:
                raise ValueError(
                    f"Cannot tokenize: {word['text']}"
                )

            word_positions.append(len(input_ids))
            input_ids.extend(token_ids)

        input_ids.append(self.tokenizer.sep_token_id)

        if len(input_ids) > MAX_LENGTH:
            raise ValueError(
                f"Comment has {len(input_ids)} tokens, "
                f"exceeding MAX_LENGTH={MAX_LENGTH}"
            )

        return words, input_ids, word_positions

    def predict(self, text):
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        if not text.strip():
            return {
                "text": text,
                "opinions": [],
            }

        words, input_ids, word_positions = (
            self.encode_text(text)
        )

        if not words:
            return {
                "text": text,
                "opinions": [],
            }

        input_tensor = torch.tensor(
            [input_ids],
            dtype=torch.long,
            device=self.device,
        )

        attention_mask = torch.ones_like(input_tensor)

        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_tensor,
                attention_mask=attention_mask,
            )

        predictions = (
            outputs.logits.argmax(dim=-1)[0].tolist()
        )

        word_labels = [
            self.id2label[predictions[pos]]
            for pos in word_positions
        ]
        
        token_debug = [
            {
                "token": word["text"],
                "start": word["start"],
                "end": word["end"],
                "label": label,
            }
            for word, label in zip(words, word_labels)
        ]

        opinions = []
        current_start = None
        current_end = None

        def close_opinion():
            nonlocal current_start, current_end

            if current_start is not None:
                opinions.append({
                    "text": text[current_start:current_end],
                    "start": current_start,
                    "end": current_end,
                })

            current_start = None
            current_end = None

        for word, label in zip(words, word_labels):
            start = word["start"]
            end = word["end"]

            if label == "B-OPINION":
                close_opinion()
                current_start = start
                current_end = end

            elif label == "I-OPINION":
                if current_start is None:
                    # Xử lý I-OPINION không có B đứng trước
                    current_start = start

                current_end = end

            else:
                close_opinion()

        close_opinion()

        return {
            "text": text,
            "opinions": opinions,
            "token_labels": token_debug,
        }


def main():
    parser = argparse.ArgumentParser(
        description="PhoBERT Opinion Span Prediction"
    )

    parser.add_argument(
        "--text",
        type=str,
        help="Comment to analyze",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=str(MODEL_PATH),
        help="Path to trained model",
    )

    args = parser.parse_args()

    predictor = OpinionPredictor(args.model)

    if args.text:
        result = predictor.predict(args.text)

        print("\n===== PREDICTION =====")
        print(json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ))

    else:
        print("\n===== INTERACTIVE MODE =====")
        print("Type 'exit' to quit.\n")

        while True:
            text = input("Comment: ").strip()

            if text.lower() in {"exit", "quit"}:
                break

            if not text:
                continue

            result = predictor.predict(text)

            print(json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
            ))
            print()


if __name__ == "__main__":
    main()
