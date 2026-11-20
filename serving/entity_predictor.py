
"""Entity Extraction Serving - PhoBERT BIO."""

import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "entity_phobert"
DICTIONARY_PATH = ROOT / "annotation" / "entity_dictionary.json"

MAX_LENGTH = 256


class EntityPredictor:
    def __init__(self, device=None):
        if not MODEL_DIR.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_DIR}")

        labels_path = MODEL_DIR / "labels.json"
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels not found: {labels_path}")

        label_data = json.loads(
            labels_path.read_text(encoding="utf-8")
        )

        self.labels = label_data["labels"]
        self.id2label = {
            int(index): label
            for index, label in enumerate(self.labels)
        }

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(MODEL_DIR),
            use_fast=False,
        )

        self.device = device or (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model = AutoModelForTokenClassification.from_pretrained(
            str(MODEL_DIR)
        )

        self.model.to(self.device)
        self.model.eval()

        self.aliases = self._load_aliases()

        print("Entity model loaded:", MODEL_DIR)
        print("Device:", self.device)

    def _load_aliases(self):
        if not DICTIONARY_PATH.exists():
            raise FileNotFoundError(
                f"Entity dictionary not found: {DICTIONARY_PATH}"
            )

        data = json.loads(
            DICTIONARY_PATH.read_text(encoding="utf-8")
        )

        aliases = set()

        for item in data.get("artists", []):
            names = [
                item.get("canonical_name", ""),
                *item.get("aliases", []),
            ]

            for alias in names:
                alias = alias.strip()

                if (
                    len(alias) >= 4
                    and re.fullmatch(r"\w+", alias, flags=re.UNICODE)
                ):
                    aliases.add(alias.casefold())

        return tuple(sorted(aliases, key=lambda x: (-len(x), x)))

    def _split_glued_artist_names(self, token):
        folded = token.casefold()

        if len(folded) != len(token):
            return [token]

        memo = {}

        def solve(pos):
            if pos == len(folded):
                return []

            if pos in memo:
                return memo[pos]

            for alias in self.aliases:
                if folded.startswith(alias, pos):
                    tail = solve(pos + len(alias))

                    if tail is not None:
                        memo[pos] = [
                            token[pos:pos + len(alias)],
                            *tail,
                        ]
                        return memo[pos]

            memo[pos] = None
            return None

        parts = solve(0)

        if parts is not None and len(parts) >= 2:
            return parts

        return [token]

    def _segment_with_offsets(self, text):
        words = []

        for match in re.finditer(r"\w+|[^\w\s]", text, re.UNICODE):
            token = match.group()
            offset = match.start()

            parts = self._split_glued_artist_names(token)

            for part in parts:
                words.append({
                    "text": part,
                    "start": offset,
                    "end": offset + len(part),
                })

                offset += len(part)

        return words

    def _encode_words(self, text):
        words = self._segment_with_offsets(text)

        input_ids = [self.tokenizer.cls_token_id]
        word_positions = []

        for word in words:
            prefix = " " if len(input_ids) > 1 else ""

            ids = self.tokenizer.encode(
                prefix + word["text"],
                add_special_tokens=False,
            )

            if not ids:
                raise ValueError(f"Cannot tokenize word: {word}")

            word_positions.append(len(input_ids))
            input_ids.extend(ids)

        input_ids.append(self.tokenizer.sep_token_id)

        if len(input_ids) > MAX_LENGTH:
            raise ValueError(
                f"Comment is too long: {len(input_ids)} tokens; "
                f"maximum is {MAX_LENGTH}"
            )

        return words, input_ids, word_positions

    def _decode_entities(self, words, predictions):
        entities = []
        current = None

        for word, label in zip(words, predictions):
            if label == "O":
                if current is not None:
                    entities.append(current)
                    current = None
                continue

            prefix, entity_type = label.split("-", 1)

            if prefix == "B" or (
                current is not None and current["type"] != entity_type
            ):
                if current is not None:
                    entities.append(current)

                current = {
                    "text": word["text"],
                    "start": word["start"],
                    "end": word["end"],
                    "type": entity_type,
                }

            elif prefix == "I":
                if current is None:
                    current = {
                        "text": word["text"],
                        "start": word["start"],
                        "end": word["end"],
                        "type": entity_type,
                    }
                else:
                    current["text"] += (
                        word["text"][
                            max(0, current["end"] - word["start"]):
                        ]
                    )
                    current["end"] = word["end"]

        if current is not None:
            entities.append(current)

        # Reconstruct text directly from original offsets.
        for entity in entities:
            entity["text"] = entity_text = (
                self._last_text[entity["start"]:entity["end"]]
            )

        return entities

    @torch.inference_mode()
    def predict(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must be a non-empty string")

        self._last_text = text

        words, input_ids, word_positions = self._encode_words(text)

        input_tensor = torch.tensor(
            [input_ids],
            dtype=torch.long,
            device=self.device,
        )

        attention_mask = torch.ones_like(input_tensor)

        outputs = self.model(
            input_ids=input_tensor,
            attention_mask=attention_mask,
        )

        logits = outputs.logits[0]

        predictions = []

        for position in word_positions:
            label_id = int(torch.argmax(logits[position]).item())
            predictions.append(self.id2label[label_id])

        entities = self._decode_entities(words, predictions)

        return {
            "text": text,
            "entities": entities,
        }


if __name__ == "__main__":
    predictor = EntityPredictor()

    samples = [
        "Team Ngáo nghễ thể hiện quá cháy trong Ngáo nghễ",
        "Màn trình diễn của nghệ sĩ rất tuyệt vời",
    ]

    for sample in samples:
        result = predictor.predict(sample)
        print(json.dumps(result, ensure_ascii=False, indent=2))
