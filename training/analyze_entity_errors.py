"""Entity PhoBERT strict span error analysis. Run: python training/analyze_entity_errors.py"""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification
from train_entity import encode_words, load_jsonl, validate_entities

ROOT = Path(__file__).resolve().parents[1]
TEST_FILE = ROOT / 'data' / 'labeled' / 'entity' / 'test.jsonl'
MODEL_DIR = ROOT / 'models' / 'entity_phobert'
OUT_DIR = ROOT / 'data' / 'analysis' / 'entity'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def predict(model, text):
    encoded = encode_words(text)
    ids = torch.tensor([encoded['input_ids']], dtype=torch.long, device=DEVICE)
    with torch.inference_mode():
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[0]
        probs = logits.softmax(dim=-1)
        labels = probs.argmax(dim=-1)
    result, current = [], None
    for word, pos in zip(encoded['words'], encoded['word_positions']):
        label = model.config.id2label[int(labels[pos])]
        if label == 'O':
            if current:
                result.append(current)
                current = None
            continue
        prefix, kind = label.split('-', 1)
        if current is None or prefix == 'B' or current['type'] != kind:
            if current:
                result.append(current)
            current = {'start': word['start'], 'end': word['end'], 'type': kind}
        else:
            current['end'] = word['end']
    if current:
        result.append(current)
    for item in result:
        item['text'] = text[item['start']:item['end']]
    return result


def overlap(a, b):
    return max(a['start'], b['start']) < min(a['end'], b['end'])


def match_errors(gold, pred):
    """One-to-one matching, exact first; diagnostic pairs never count as TP."""
    used_g, used_p, matches = set(), set(), []
    rules = [
        ('TP', lambda g, p: g['start'] == p['start'] and g['end'] == p['end'] and g['type'] == p['type']),
        ('TYPE_ERROR', lambda g, p: g['start'] == p['start'] and g['end'] == p['end'] and g['type'] != p['type']),
        ('BOUNDARY_ERROR', lambda g, p: g['type'] == p['type'] and overlap(g, p)),
        ('TYPE_AND_BOUNDARY_ERROR', lambda g, p: g['type'] != p['type'] and overlap(g, p)),
    ]
    for kind, rule in rules:
        for gi, g in enumerate(gold):
            if gi in used_g:
                continue
            candidates = [(pi, p) for pi, p in enumerate(pred) if pi not in used_p and rule(g, p)]
            if candidates:
                pi, p = max(candidates, key=lambda x: min(g['end'], x[1]['end']) - max(g['start'], x[1]['start']))
                used_g.add(gi)
                used_p.add(pi)
                matches.append((kind, g, p))
    for gi, g in enumerate(gold):
        if gi not in used_g:
            matches.append(('FN', g, None))
    for pi, p in enumerate(pred):
        if pi not in used_p:
            matches.append(('FP', None, p))
    return matches


def span_str(items):
    return ' | '.join(f"{x['text']} ({x['type']}) [{x['start']}:{x['end']}]" for x in items)


def write_csv(path, fields, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = AutoModelForTokenClassification.from_pretrained(str(MODEL_DIR)).to(DEVICE)
    model.eval()
    records = load_jsonl(TEST_FILE)
    kinds = [str(x) for x in ['ARTIST', 'TEAM', 'PERFORMANCE', 'SHOW']]
    counts = defaultdict(Counter)
    errors = Counter()
    confusions = Counter()
    comments, spans = [], []

    for index, record in enumerate(records):
        text = record['text']
        cid = record.get('comment_id', f'row_{index}')
        gold = validate_entities(record)
        pred = predict(model, text)
        matches = match_errors(gold, pred)
        local = Counter()
        for kind, g, p in matches:
            local[kind] += 1
            errors[kind] += 1
            if kind == 'TP':
                counts[g['type']]['tp'] += 1
            else:
                if g is not None:
                    counts[g['type']]['fn'] += 1
                if p is not None:
                    counts[p['type']]['fp'] += 1
            if kind in ('TYPE_ERROR', 'TYPE_AND_BOUNDARY_ERROR'):
                confusions[(g['type'], p['type'])] += 1
            spans.append({
                'comment_id': cid, 'text': text,
                'gold': span_str([g]) if g else '',
                'predicted': span_str([p]) if p else '',
                'error_type': kind,
            })
        if not gold and not pred:
            category = 'TRUE_NEGATIVE'
        elif all(x == 'TP' for x in local):
            category = 'CORRECT'
        elif any('BOUNDARY' in x for x in local):
            category = 'BOUNDARY_ERROR'
        elif 'TYPE_ERROR' in local:
            category = 'TYPE_ERROR'
        elif local['FP'] and local['FN']:
            category = 'FP_AND_FN'
        elif local['FP']:
            category = 'FALSE_POSITIVE'
        else:
            category = 'FALSE_NEGATIVE'
        comments.append({
            'comment_id': cid, 'text': text,
            'gold_entities': span_str(gold), 'predicted_entities': span_str(pred),
            'tp': local['TP'],
            'fp': len(pred) - local['TP'],
            'fn': len(gold) - local['TP'],
            'error_type': category,
        })

    def metrics(tp, fp, fn):
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': precision,
                'recall': recall, 'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.0}

    per_type = {kind: metrics(counts[kind]['tp'], counts[kind]['fp'], counts[kind]['fn']) for kind in kinds}
    overall = metrics(sum(v['tp'] for v in per_type.values()),
                      sum(v['fp'] for v in per_type.values()),
                      sum(v['fn'] for v in per_type.values()))
    summary = {
        'total_comments': len(records),
        'gold_spans': sum(len(r.get('entities', [])) for r in records),
        'predicted_spans': sum(int(r['tp']) + int(r['fp']) for r in comments),
        **overall, 'per_type': per_type,
        'pair_error_counts': dict(errors),
        'comment_error_counts': dict(Counter(r['error_type'] for r in comments)),
        'type_confusions': [{'gold_type': a, 'predicted_type': b, 'count': n}
                            for (a, b), n in sorted(confusions.items())],
    }
    write_csv(OUT_DIR / 'comment_analysis.csv',
              ['comment_id', 'text', 'gold_entities', 'predicted_entities', 'tp', 'fp', 'fn', 'error_type'], comments)
    write_csv(OUT_DIR / 'span_errors.csv',
              ['comment_id', 'text', 'gold', 'predicted', 'error_type'], spans)
    write_csv(OUT_DIR / 'confusion_matrix.csv',
              ['gold_type', 'predicted_type', 'count'], summary['type_confusions'])
    (OUT_DIR / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\n===== ENTITY ERROR ANALYSIS =====')
    print(f'Device: {DEVICE} | Comments: {len(records)}')
    print('Overall:', overall)
    for kind, result in per_type.items():
        print(f'{kind:12s} TP={result["tp"]:3d} FP={result["fp"]:3d} FN={result["fn"]:3d} F1={result["f1"]:.4f}')
    print('Error types:', dict(errors))
    print('Files saved to:', OUT_DIR)


if __name__ == '__main__':
    main()
