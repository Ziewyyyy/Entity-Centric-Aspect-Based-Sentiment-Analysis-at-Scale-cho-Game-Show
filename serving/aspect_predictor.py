
"""
Aspect Classification Serving V2 - PhoBERT

Model:
    models/aspect_phobert_v2

Features:
- Vietnamese word segmentation
- Opinion span markers
- 12 aspect classes
- CUDA / CPU fallback
- Batch prediction
"""

import json
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from underthesea import word_tokenize


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT / "models" / "aspect_phobert_v2" / "checkpoint-2352"

MAX_LENGTH = 256

MARK_START = "<opinion>"
MARK_END = "</opinion>"

LABELS = [
    "VOCAL",
    "DANCE",
    "PERFORMANCE",
    "VISUAL",
    "PERSONALITY",
    "INTERACTION",
    "CONTENT",
    "PRODUCTION",
    "EDITING",
    "GENERAL",
    "POPULARITY",
    "AWARD_RESULT",
]


# ============================================================
# ASPECT PREDICTOR
# ============================================================

class AspectPredictor:

    def __init__(self, device=None):

        # ----------------------------------------------------
        # Check model directory
        # ----------------------------------------------------

        if not MODEL_DIR.exists():
            raise FileNotFoundError(
                f"Aspect model not found: {MODEL_DIR}"
            )

        if not (MODEL_DIR / "config.json").exists():
            raise FileNotFoundError(
                f"Model config not found: {MODEL_DIR}"
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
        # Load tokenizer
        # ----------------------------------------------------

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR,
            use_fast=False,
        )

        # ----------------------------------------------------
        # Load model
        # ----------------------------------------------------

        self.model = (
            AutoModelForSequenceClassification
            .from_pretrained(MODEL_DIR)
        )

        self.model.to(self.device)
        self.model.eval()

        # ----------------------------------------------------
        # Load labels from model config
        # ----------------------------------------------------

        config_labels = self.model.config.id2label

        self.id2label = {
            int(index): label
            for index, label in config_labels.items()
        }

        self.label2id = {
            label: index
            for index, label in self.id2label.items()
        }

        # Validate labels
        if set(self.id2label.values()) != set(LABELS):
            raise ValueError(
                "Aspect labels do not match training labels.\n"
                f"Expected: {LABELS}\n"
                f"Received: {self.id2label}"
            )

        # ----------------------------------------------------
        # Validate special markers
        # ----------------------------------------------------

        self.marker_start_id = (
            self.tokenizer.convert_tokens_to_ids(
                MARK_START
            )
        )

        self.marker_end_id = (
            self.tokenizer.convert_tokens_to_ids(
                MARK_END
            )
        )

        if (
            self.marker_start_id == self.tokenizer.unk_token_id
            or self.marker_end_id == self.tokenizer.unk_token_id
        ):
            raise ValueError(
                "Opinion markers are missing from tokenizer."
            )

        print(
            f"Aspect model loaded: {MODEL_DIR}"
        )

        print("Device:", self.device)

        print("Number of aspects:", len(self.id2label))

        print("Labels:", self.id2label)

    # ========================================================
    # VIETNAMESE WORD SEGMENTATION
    # ========================================================

    @staticmethod
    def segment(text):

        if not text.strip():
            return ""

        return word_tokenize(
            text,
            format="text",
        )

    # ========================================================
    # PREPARE INPUT
    # ========================================================

    def prepare_input(
        self,
        text,
        opinion,
        start=None,
        end=None,
    ):

        if not isinstance(text, str):
            raise TypeError(
                "text must be a string"
            )

        if not isinstance(opinion, str):
            raise TypeError(
                "opinion must be a string"
            )

        if not text.strip():
            raise ValueError(
                "Input text cannot be empty"
            )

        if not opinion.strip():
            raise ValueError(
                "Opinion cannot be empty"
            )

        # ----------------------------------------------------
        # Locate opinion if offsets are not provided
        # ----------------------------------------------------

        if start is None and end is None:

            start = text.find(opinion)

            if start == -1:
                raise ValueError(
                    f"Opinion not found in text: {opinion}"
                )

            end = start + len(opinion)

        elif start is None or end is None:

            raise ValueError(
                "start and end must be provided together"
            )

        # ----------------------------------------------------
        # Validate offsets
        # ----------------------------------------------------

        if not isinstance(start, int) or not isinstance(end, int):
            raise TypeError(
                "start and end must be integers"
            )

        if not (0 <= start < end <= len(text)):
            raise ValueError(
                f"Invalid opinion offsets: {start}, {end}"
            )

        actual_opinion = text[start:end]

        if actual_opinion != opinion:
            raise ValueError(
                "Opinion offset mismatch.\n"
                f"Expected: {opinion}\n"
                f"Actual: {actual_opinion}"
            )

        # ----------------------------------------------------
        # Segment context
        # ----------------------------------------------------

        before = self.segment(
            text[:start].strip()
        )

        target = self.segment(
            text[start:end].strip()
        )

        after = self.segment(
            text[end:].strip()
        )

        segmented_opinion = self.segment(
            opinion
        )

        # ----------------------------------------------------
        # Add opinion markers
        # ----------------------------------------------------

        context = " ".join(
            item
            for item in (
                before,
                MARK_START,
                target,
                MARK_END,
                after,
            )
            if item
        )

        # Same format as training
        model_input = (
            f"{context} </s> {segmented_opinion}"
        )

        return model_input

    # ========================================================
    # TOKENIZE INPUT
    # ========================================================

    def encode_input(self, text):

        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        return {
            key: value.to(self.device)
            for key, value in encoded.items()
        }

    # ========================================================
    # PREDICT ONE OPINION
    # ========================================================

    @torch.inference_mode()
    def predict(
        self,
        text,
        opinion,
        start=None,
        end=None,
    ):

        model_input = self.prepare_input(
            text=text,
            opinion=opinion,
            start=start,
            end=end,
        )

        encoded = self.encode_input(
            model_input
        )

        outputs = self.model(
            **encoded
        )

        probabilities = torch.softmax(
            outputs.logits.float(),
            dim=-1,
        )[0]

        predicted_id = int(
            torch.argmax(probabilities).item()
        )

        predicted_label = self.id2label[
            predicted_id
        ]

        scores = {
            self.id2label[index]: round(
                float(probability.item()),
                6,
            )
            for index, probability in enumerate(
                probabilities
            )
        }

        return {
            "text": text,
            "opinion": opinion,
            "aspect": predicted_label,
            "confidence": round(
                float(probabilities[predicted_id].item()),
                6,
            ),
            "scores": scores,
        }

    # ========================================================
    # PREDICT MULTIPLE OPINIONS
    # ========================================================

    def predict_many(self, text, opinions):

        results = []

        for opinion in opinions:

            if isinstance(opinion, dict):

                result = self.predict(
                    text=text,
                    opinion=opinion["text"],
                    start=opinion.get("start"),
                    end=opinion.get("end"),
                )

            else:

                result = self.predict(
                    text=text,
                    opinion=opinion,
                )

            results.append(result)

        return {
            "text": text,
            "aspects": results,
        }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    predictor = AspectPredictor()

    text = (
        "Team Ngáo nghễ thể hiện quá cháy "
        "trong phần trình diễn. "
        "Giọng hát rất tuyệt vời."
    )

    test_opinions = [
        {
            "text": "quá cháy",
            "start": 24,
            "end": 32,
        },
        {
            "text": "rất tuyệt vời",
            "start": 66,
            "end": 79,
        },
    ]

    result = predictor.predict_many(
        text=text,
        opinions=test_opinions,
    )

    print("\n" + "=" * 60)
    print("ASPECT PREDICTION TEST")
    print("=" * 60)

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )
