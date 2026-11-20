
import json

from serving.entity_predictor import EntityPredictor
from serving.opinion_predictor import OpinionPredictor
from serving.relation_predictor import RelationPredictor
from serving.aspect_predictor import AspectPredictor
from serving.sentiment_predictor import SentimentPredictor


class SentimentAnalysisPipeline:
    """
    End-to-end Entity-Centric Aspect-Based Sentiment Analysis Pipeline.
    """

    def __init__(self):
        print("Loading Entity Predictor...")
        self.entity_predictor = EntityPredictor()

        print("Loading Opinion Predictor...")
        self.opinion_predictor = OpinionPredictor()

        print("Loading Relation Predictor...")
        self.relation_predictor = RelationPredictor()

        print("Loading Aspect Predictor...")
        self.aspect_predictor = AspectPredictor()

        print("Loading Sentiment Predictor...")
        self.sentiment_predictor = SentimentPredictor()

        print("All models loaded successfully.")

    def predict(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input comment must be a non-empty string.")

        # 1. Entity Extraction
        entity_result = self.entity_predictor.predict(text)
        entities = entity_result["entities"]

        # 2. Opinion Extraction
        opinion_result = self.opinion_predictor.predict(text)
        opinions = opinion_result["opinions"]

        # 3. Entity-Opinion Relation Extraction
        relation_result = self.relation_predictor.predict(
            text=text,
            entities=entities,
            opinions=opinions,
        )

        # 4. Aspect and Sentiment Classification
        enriched_opinions = []

        for opinion in opinions:
            aspect_result = self.aspect_predictor.predict(
                text=text,
                opinion=opinion["text"],
                start=opinion["start"],
                end=opinion["end"],
            )

            aspect = aspect_result["aspect"]

            sentiment_result = self.sentiment_predictor.predict(
                text=text,
                opinion=opinion["text"],
                aspect=aspect,
                start=opinion["start"],
                end=opinion["end"],
            )

            enriched_opinions.append({
                **opinion,
                "aspect": aspect,
                "aspect_confidence": aspect_result["confidence"],
                "sentiment": sentiment_result["sentiment"],
                "sentiment_confidence": sentiment_result["confidence"],
            })

        # 5. Group opinions by their linked entities
        entity_results = []

        for entity_index, entity in enumerate(entities):
            linked_opinions = []

            for pair in relation_result["pairs"]:
                if (
                    pair["relation"] == "HAS_OPINION"
                    and pair["entity_index"] == entity_index
                ):
                    opinion_index = pair["opinion_index"]

                    if opinion_index < len(enriched_opinions):
                        opinion_data = enriched_opinions[opinion_index]

                        linked_opinions.append({
                            "text": opinion_data["text"],
                            "start": opinion_data["start"],
                            "end": opinion_data["end"],
                            "aspect": opinion_data["aspect"],
                            "aspect_confidence": opinion_data["aspect_confidence"],
                            "sentiment": opinion_data["sentiment"],
                            "sentiment_confidence": opinion_data["sentiment_confidence"],
                            "relation_confidence": pair["probability"],
                        })

            entity_results.append({
                "entity": entity["text"],
                "entity_type": entity["type"],
                "start": entity["start"],
                "end": entity["end"],
                "opinions": linked_opinions,
            })

        # 6. Return complete result
        return {
            "text": text,
            "entities": entity_results,
            "opinions": enriched_opinions,
            "relations": relation_result["relations"],
        }


def main():
    print("=" * 60)
    print("ENTITY-CENTRIC ASPECT-BASED SENTIMENT ANALYSIS")
    print("=" * 60)

    pipeline = SentimentAnalysisPipeline()

    while True:
        print("\nEnter a comment (type 'exit' to quit):")
        text = input("> ").strip()

        if text.lower() in ["exit", "quit"]:
            print("Exiting...")
            break

        if not text:
            print("Please enter a non-empty comment.")
            continue

        try:
            result = pipeline.predict(text)

            print("\nANALYSIS RESULT")
            print("-" * 60)
            print(json.dumps(
                result,
                ensure_ascii=False,
                indent=4,
            ))

        except Exception as error:
            print(f"\nPrediction error: {error}")


if __name__ == "__main__":
    main()
