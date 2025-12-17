import json 
# from rich import print
from utils.span import SpanSelector
from utils.processing import build_conflict_matrix
from utils.generator import T5Generator
selector = SpanSelector()
filler = T5Generator()


with open('datasets/yelp/yelp_70_300.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
for item in data:
    text = item['text']
    spans = selector.get_spans(text)
    # matrix = build_conflict_matrix(spans)
    # print(matrix)
    for span in spans:
        candidates = filler.generate_candidates(text, span['text'])
        print(span['text'])
        print(candidates)
        break
    break
