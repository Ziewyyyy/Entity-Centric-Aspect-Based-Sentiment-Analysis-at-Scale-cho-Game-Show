import torch
from pathlib import Path
from transformers import AutoModelForTokenClassification

from train_entity import encode_words

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / 'models' / 'entity_phobert'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = AutoModelForTokenClassification.from_pretrained(str(MODEL_DIR))
model.to(DEVICE)
model.eval()
id2label = model.config.id2label


def predict_entities(text):
    encoded = encode_words(text)
    words = encoded['words']
    input_ids = encoded['input_ids']
    positions = encoded['word_positions']

    inputs = {
        'input_ids': torch.tensor([input_ids], dtype=torch.long, device=DEVICE),
        'attention_mask': torch.ones((1, len(input_ids)), dtype=torch.long, device=DEVICE),
    }
    with torch.no_grad():
        logits = model(**inputs).logits[0]
        probabilities = torch.softmax(logits, dim=-1)
        predicted_ids = probabilities.argmax(dim=-1)

    entities = []
    current = None
    for word, position in zip(words, positions):
        predicted_id = int(predicted_ids[position])
        label = id2label[predicted_id]
        confidence = float(probabilities[position, predicted_id])

        if label == 'O':
            if current is not None:
                entities.append(current)
                current = None
            continue

        prefix, entity_type = label.split('-', 1)
        if prefix == 'B' or current is None or current['type'] != entity_type:
            if current is not None:
                entities.append(current)
            current = {
                'text': word['text'],
                'type': entity_type,
                'start': word['start'],
                'end': word['end'],
                'scores': [confidence],
            }
        else:
            current['end'] = word['end']
            current['text'] = text[current['start']:current['end']]
            current['scores'].append(confidence)

    if current is not None:
        entities.append(current)

    results = []
    for entity in entities:
        scores = entity.pop('scores')
        results.append({
            'text': entity['text'],
            'type': entity['type'],
            'start': entity['start'],
            'end': entity['end'],
            'confidence': round(sum(scores) / len(scores), 4),
        })
    return results


if __name__ == '__main__':
    print(f'Device: {DEVICE}')
    print('PhoBERT Entity Recognition')
    print("Nhập 'exit' để thoát.")
    while True:
        text = input('\nComment: ').strip()
        if text.lower() == 'exit':
            break
        if not text:
            continue
        try:
            entities = predict_entities(text)
            if not entities:
                print('Không tìm thấy entity.')
            for entity in entities:
                print(
                    f"{entity['text']!r} -> {entity['type']} "
                    f"[{entity['start']}, {entity['end']}] "
                    f"confidence={entity['confidence']:.4f}"
                )
        except Exception as e:
            print(f'Error: {e}')
