
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from serving import text_utils as tu

MODEL_DIR = ROOT / "models" / "relation_phobert"

LABELS = ["NO_RELATION", "HAS_OPINION"]
MAX_LENGTH = 256


class RelationPredictor:

    def __init__(self, model_dir=MODEL_DIR):

        self.model_dir = Path(model_dir)

        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Model directory not found: {self.model_dir}"
            )

        config_path = self.model_dir / "relation_config.json"

        if not config_path.exists():
            raise FileNotFoundError(
                f"Missing relation_config.json: {config_path}"
            )

        with config_path.open("r", encoding="utf-8") as f:
            self.relation_config = json.load(f)

        self.threshold = float(
            self.relation_config["threshold"]
        )

        self.max_length = int(
            self.relation_config.get("max_length", MAX_LENGTH)
        )

        self.markers = self.relation_config.get(
            "markers",
            tu.RELATION_MARKERS,
        )

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir),
            use_fast=False,
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_dir)
        )

        self.model.to(self.device)
        self.model.eval()

        config_labels = self.model.config.id2label

        self.id2label = {
            int(index): label
            for index, label in config_labels.items()
        }

        if set(self.id2label.values()) != set(LABELS):
            raise ValueError(
                f"Unexpected relation labels: {self.id2label}"
            )

        for marker in self.markers:
            marker_id = self.tokenizer.convert_tokens_to_ids(marker)

            if (
                marker_id is None
                or marker_id == self.tokenizer.unk_token_id
            ):
                raise ValueError(
                    f"Missing special token: {marker}"
                )

        print("=" * 60)
        print("RELATION MODEL LOADED")
        print("=" * 60)
        print("Model:", self.model_dir)
        print("Device:", self.device)
        print("Labels:", self.id2label)
        print("Threshold:", self.threshold)
        print("Max length:", self.max_length)

    def build_input(self, text, entity, opinion):
        """
        Sử dụng đúng hàm build_relation_input lúc training.
        """
        return tu.build_relation_input(
            text,
            entity,
            opinion,
        )

    def encode_input(self, model_input):

        encoded = self.tokenizer(
            model_input,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Training loại các cặp bị cắt mất marker.
        input_ids = encoded["input_ids"][0].tolist()

        marker_ids = [
            self.tokenizer.convert_tokens_to_ids(marker)
            for marker in self.markers
        ]

        missing = [
            marker
            for marker, marker_id in zip(
                self.markers,
                marker_ids,
            )
            if marker_id not in input_ids
        ]

        if missing:
            raise ValueError(
                "Input was truncated and lost markers: "
                + ", ".join(missing)
            )

        return {
            key: value.to(self.device)
            for key, value in encoded.items()
        }

    @torch.inference_mode()
    def predict_pair(self, text, entity, opinion):

        model_input = self.build_input(
            text,
            entity,
            opinion,
        )

        inputs = self.encode_input(model_input)

        outputs = self.model(**inputs)

        probabilities = F.softmax(
            outputs.logits,
            dim=-1,
        )[0]

        positive_probability = float(
            probabilities[1].item()
        )

        predicted_label = (
            "HAS_OPINION"
            if positive_probability >= self.threshold
            else "NO_RELATION"
        )

        return {
            "entity": entity["text"],
            "entity_type": entity.get("type", "UNKNOWN"),
            "opinion": opinion["text"],
            "relation": predicted_label,
            "probability": round(positive_probability, 6),
            "threshold": self.threshold,
            "scores": {
                self.id2label[i]: round(
                    float(probabilities[i].item()),
                    6,
                )
                for i in range(len(self.id2label))
            },
        }

    def predict(self, text, entities, opinions):

        pairs = []
        grouped = {}

        for opinion_index, opinion in enumerate(opinions):

            opinion_results = []

            for entity_index, entity in enumerate(entities):

                result = self.predict_pair(
                    text,
                    entity,
                    opinion,
                )

                result["entity_index"] = entity_index
                result["opinion_index"] = opinion_index

                pairs.append(result)

                if result["relation"] == "HAS_OPINION":
                    opinion_results.append(result)

            grouped[str(opinion_index)] = {
                "opinion": opinion["text"],
                "linked_entities": opinion_results,
            }

        return {
            "text": text,
            "threshold": self.threshold,
            "entities": entities,
            "opinions": opinions,
            "pairs": pairs,
            "relations": [
                {
                    "entity": result["entity"],
                    "entity_type": result["entity_type"],
                    "opinion": result["opinion"],
                    "probability": result["probability"],
                }
                for result in pairs
                if result["relation"] == "HAS_OPINION"
            ],
            "grouped_by_opinion": grouped,
        }


if __name__ == "__main__":

    import json

    predictor = RelationPredictor()

    text = (
        "Team Ngáo nghễ thể hiện quá cháy "
        "trong phần trình diễn. "
        "Đội Mặt Trời nhảy cực kỳ đẹp."
    )

    entities = [
        {
            "text": "Team Ngáo nghễ",
            "start": text.index("Team Ngáo nghễ"),
            "end": text.index("Team Ngáo nghễ") + len("Team Ngáo nghễ"),
            "type": "TEAM",
        },
        {
            "text": "Đội Mặt Trời",
            "start": text.index("Đội Mặt Trời"),
            "end": text.index("Đội Mặt Trời") + len("Đội Mặt Trời"),
            "type": "TEAM",
        },
    ]

    opinions = [
        {
            "text": "quá cháy",
            "start": text.index("quá cháy"),
            "end": text.index("quá cháy") + len("quá cháy"),
        },
        {
            "text": "cực kỳ đẹp",
            "start": text.index("cực kỳ đẹp"),
            "end": text.index("cực kỳ đẹp") + len("cực kỳ đẹp"),
        },
    ]

    result = predictor.predict(
        text=text,
        entities=entities,
        opinions=opinions,
    )

    print("\n" + "=" * 60)
    print("RELATION PREDICTION TEST")
    print("=" * 60)

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )
