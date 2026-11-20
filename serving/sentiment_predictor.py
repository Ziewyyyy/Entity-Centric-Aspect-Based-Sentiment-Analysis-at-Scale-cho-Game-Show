
import torch
import torch.nn.functional as F

from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


ROOT = Path(__file__).resolve().parents[1]

# Model và tokenizer nằm trực tiếp trong thư mục này
MODEL_DIR = ROOT / "models" / "sentiment_phobert"

LABELS = ["NEG", "NEU", "POS"]

ASPECTS = {
    "GENERAL",
    "VOCAL",
    "CONTENT",
    "VISUAL",
    "POPULARITY",
    "PRODUCTION",
    "PERFORMANCE",
    "DANCE",
    "INTERACTION",
    "PERSONALITY",
    "AWARD_RESULT",
    "EDITING",
}

MAX_LENGTH = 192

OP_START = "<op>"
OP_END = "</op>"


class SentimentPredictor:

    def __init__(self, model_dir=MODEL_DIR):

        self.model_dir = Path(model_dir)

        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Model directory not found: {self.model_dir}"
            )

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir),
            use_fast=False,
        )

        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_dir)
        )

        self.model.to(self.device)
        self.model.eval()

        # Read labels from model config
        config_labels = self.model.config.id2label

        self.id2label = {
            int(index): label
            for index, label in config_labels.items()
        }

        self.label2id = {
            label: index
            for index, label in self.id2label.items()
        }

        if set(self.id2label.values()) != set(LABELS):
            raise ValueError(
                f"Unexpected sentiment labels: {self.id2label}"
            )

        # Verify special tokens
        for token in (OP_START, OP_END):
            token_id = self.tokenizer.convert_tokens_to_ids(token)

            if (
                token_id is None
                or token_id == self.tokenizer.unk_token_id
            ):
                raise ValueError(
                    f"Missing special token: {token}"
                )

        print("=" * 60)
        print("SENTIMENT MODEL LOADED")
        print("=" * 60)
        print("Model:", self.model_dir)
        print("Device:", self.device)
        print("Labels:", self.id2label)

    def build_input(
        self,
        text,
        opinion,
        aspect,
        start=None,
        end=None,
    ):
        """
        Build input exactly like training.

        Example:
        Text: Isaac hát hay nhưng nhảy chưa tốt.
        Opinion: hay
        Aspect: VOCAL

        Input:
        Isaac hát <op> hay </op> nhưng nhảy chưa tốt.
        </s> Aspect: VOCAL
        """

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text must not be empty.")

        if not isinstance(opinion, str) or not opinion:
            raise ValueError("Opinion must not be empty.")

        if aspect not in ASPECTS:
            raise ValueError(f"Invalid aspect: {aspect}")

        # Find opinion automatically if offsets are omitted
        if start is None and end is None:
            start = text.find(opinion)

            if start == -1:
                raise ValueError(
                    f"Opinion '{opinion}' not found in text."
                )

            end = start + len(opinion)

        elif start is None or end is None:
            raise ValueError(
                "start and end must both be provided."
            )

        # Validate offsets
        if not (
            isinstance(start, int)
            and isinstance(end, int)
            and 0 <= start < end <= len(text)
        ):
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

        # Same input construction as training
        marked_text = (
            text[:start]
            + f" {OP_START} "
            + text[start:end]
            + f" {OP_END} "
            + text[end:]
        )

        model_input = (
            marked_text
            + " </s> Aspect: "
            + aspect
        )

        return model_input

    def encode_input(self, model_input):

        encoded = self.tokenizer(
            model_input,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        return {
            key: value.to(self.device)
            for key, value in encoded.items()
        }

    @torch.inference_mode()
    def predict(
        self,
        text,
        opinion,
        aspect,
        start=None,
        end=None,
    ):

        model_input = self.build_input(
            text=text,
            opinion=opinion,
            aspect=aspect,
            start=start,
            end=end,
        )

        inputs = self.encode_input(model_input)

        outputs = self.model(**inputs)

        probabilities = F.softmax(
            outputs.logits,
            dim=-1,
        )[0]

        predicted_id = int(
            torch.argmax(probabilities).item()
        )

        predicted_sentiment = self.id2label[predicted_id]

        scores = {
            self.id2label[index]: round(
                float(probabilities[index].item()),
                6,
            )
            for index in range(len(self.id2label))
        }

        return {
            "text": text,
            "opinion": opinion,
            "aspect": aspect,
            "sentiment": predicted_sentiment,
            "confidence": round(
                float(probabilities[predicted_id].item()),
                6,
            ),
            "scores": scores,
        }

    def predict_many(self, text, opinions):

        results = []

        for item in opinions:

            if isinstance(item, str):
                opinion = item
                aspect = "GENERAL"
                start = None
                end = None

            else:
                opinion = item["text"]
                aspect = item["aspect"]

                start = item.get("start")
                end = item.get("end")

            result = self.predict(
                text=text,
                opinion=opinion,
                aspect=aspect,
                start=start,
                end=end,
            )

            results.append(result)

        return {
            "text": text,
            "sentiments": results,
        }


if __name__ == "__main__":

    import json

    predictor = SentimentPredictor()

    text = (
        "Team Ngáo nghễ thể hiện quá cháy "
        "trong phần trình diễn. "
        "Giọng hát rất tuyệt vời."
    )

    test_opinions = [
        {
            "text": "quá cháy",
            "start": text.index("quá cháy"),
            "end": text.index("quá cháy") + len("quá cháy"),
            "aspect": "PERFORMANCE",
        },
        {
            "text": "rất tuyệt vời",
            "start": text.index("rất tuyệt vời"),
            "end": text.index("rất tuyệt vời") + len("rất tuyệt vời"),
            "aspect": "VOCAL",
        },
    ]

    result = predictor.predict_many(
        text=text,
        opinions=test_opinions,
    )

    print("\n" + "=" * 60)
    print("SENTIMENT PREDICTION TEST")
    print("=" * 60)

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )
