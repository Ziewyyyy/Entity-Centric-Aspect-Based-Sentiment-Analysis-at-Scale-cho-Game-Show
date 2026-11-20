
"""
Opinion Extraction Serving - PhoBERT BIO
Model: models/opinion_phobert/checkpoint-330
"""

import json
import re
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
)


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = (
    ROOT
    / "models"
    / "opinion_phobert"
    / "checkpoint-330"
)

MAX_LENGTH = 256

TOKEN_PATTERN = re.compile(
    r"\w+|[^\w\s]",
    re.UNICODE,
)


# ============================================================
# OPINION PREDICTOR
# ============================================================

class OpinionPredictor:

    def __init__(self, device=None):

        # ----------------------------------------------------
        # Check model directory
        # ----------------------------------------------------

        if not MODEL_DIR.exists():
            raise FileNotFoundError(
                f"Opinion model not found: {MODEL_DIR}"
            )

        if not (MODEL_DIR / "model.safetensors").exists():
            raise FileNotFoundError(
                f"Model weights not found: {MODEL_DIR}"
            )

        # ----------------------------------------------------
        # Load tokenizer
        # ----------------------------------------------------

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR,
            use_fast=False,
        )

        # ----------------------------------------------------
        # Device
        # ----------------------------------------------------

        self.device = torch.device(
            device or (
                "cuda:0"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        # ----------------------------------------------------
        # Load model
        # ----------------------------------------------------

        self.model = (
            AutoModelForTokenClassification
            .from_pretrained(MODEL_DIR)
        )

        # Load labels directly from config.json
        self.id2label = {
            int(index): label
            for index, label
            in self.model.config.id2label.items()
        }

        # Validate labels
        expected_labels = {
            "O",
            "B-OPINION",
            "I-OPINION",
        }

        if set(self.id2label.values()) != expected_labels:
            raise ValueError(
                f"Unexpected model labels: {self.id2label}"
            )

        self.model.to(self.device)
        self.model.eval()

        print(
            f"Opinion model loaded: {MODEL_DIR}"
        )

        print("Device:", self.device)

        print("Labels:", self.id2label)

    # ========================================================
    # TOKENIZE WITH CHARACTER OFFSETS
    # ========================================================

    @staticmethod
    def tokenize_with_offsets(text):

        return [
            {
                "text": match.group(),
                "start": match.start(),
                "end": match.end(),
            }
            for match in TOKEN_PATTERN.finditer(text)
        ]

    # ========================================================
    # ENCODE WORDS
    # ========================================================

    def encode_words(self, words):

        input_ids = [
            self.tokenizer.cls_token_id
        ]

        word_positions = []

        for word in words:

            token_ids = self.tokenizer.encode(
                word["text"],
                add_special_tokens=False,
            )

            if not token_ids:
                raise ValueError(
                    f"Tokenizer returned no tokens: {word}"
                )

            # Position of the first subword
            word_positions.append(
                len(input_ids)
            )

            input_ids.extend(token_ids)

        # Add SEP token
        input_ids.append(
            self.tokenizer.sep_token_id
        )

        if len(input_ids) > MAX_LENGTH:
            raise ValueError(
                f"Sequence too long: {len(input_ids)} tokens. "
                f"Maximum allowed: {MAX_LENGTH}"
            )

        attention_mask = [
            1
        ] * len(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "word_positions": word_positions,
        }

    # ========================================================
    # DECODE BIO PREDICTIONS
    # ========================================================

    @staticmethod
    def decode_entities(text, words, predictions):

        opinions = []

        current = None

        def flush():

            nonlocal current

            if current is not None:

                current["text"] = text[
                    current["start"]:current["end"]
                ]

                opinions.append(current)

                current = None

        for word, label in zip(
            words,
            predictions,
        ):

            if label == "O":
                flush()
                continue

            prefix, entity_type = label.split("-", 1)

            if entity_type != "OPINION":
                flush()
                continue

            if (
                prefix == "B"
                or current is None
                or current["type"] != entity_type
            ):

                flush()

                current = {
                    "start": word["start"],
                    "end": word["end"],
                    "type": entity_type,
                }

            else:

                # Continue current BIO entity
                current["end"] = word["end"]

        flush()

        return opinions

    # ========================================================
    # PREDICT
    # ========================================================

    @torch.inference_mode()
    def predict(self, text):

        if not isinstance(text, str):
            raise TypeError(
                "text must be a string"
            )

        if not text.strip():
            raise ValueError(
                "Input text cannot be empty"
            )

        # ----------------------------------------------------
        # Tokenize
        # ----------------------------------------------------

        words = self.tokenize_with_offsets(text)

        if not words:
            return {
                "text": text,
                "opinions": [],
            }

        # ----------------------------------------------------
        # Encode
        # ----------------------------------------------------

        encoded = self.encode_words(words)

        input_ids = torch.tensor(
            [encoded["input_ids"]],
            dtype=torch.long,
            device=self.device,
        )

        attention_mask = torch.tensor(
            [encoded["attention_mask"]],
            dtype=torch.long,
            device=self.device,
        )

        # ----------------------------------------------------
        # Model inference
        # ----------------------------------------------------

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        predictions = torch.argmax(
            outputs.logits,
            dim=-1,
        )[0]

        # ----------------------------------------------------
        # Get prediction at first subword of each word
        # ----------------------------------------------------

        word_predictions = []

        for position in encoded["word_positions"]:

            pred_id = int(
                predictions[position].item()
            )

            word_predictions.append(
                self.id2label[pred_id]
            )

        # ----------------------------------------------------
        # Decode BIO
        # ----------------------------------------------------

        opinions = self.decode_entities(
            text,
            words,
            word_predictions,
        )

        return {
            "text": text,
            "opinions": opinions,
        }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    predictor = OpinionPredictor()

    test_sentences = [

        "Team Ngáo nghễ thể hiện quá cháy trong Ngáo nghễ",

        "Màn trình diễn của nghệ sĩ rất tuyệt vời",

        "Giọng hát quá hay và cảm xúc",

        "Phần trình diễn này thật sự xuất sắc",

        "Tôi không thích tiết mục này",

    ]

    print("\n" + "=" * 60)
    print("OPINION PREDICTION TEST")
    print("=" * 60)

    for sentence in test_sentences:

        result = predictor.predict(sentence)

        print("\nInput:", sentence)

        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
            )
        )
