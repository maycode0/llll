from transformers import BertTokenizer,BertForSequenceClassification
from transformers import BartTokenizer, BartForConditionalGeneration
from utils.generator import T5Generator
from utils.span import SpanSelector
import torch
span_selector = SpanSelector()

fp = 'models/bert-base-uncased-yelp-polarity'
tokenizer = BertTokenizer.from_pretrained(fp)
model = BertForSequenceClassification.from_pretrained(fp).to('cuda')

# 1. 加载模型和分词器
bart_fp = 'models/bart-base'
bart_tokenizer = BartTokenizer.from_pretrained(bart_fp)
bart = BartForConditionalGeneration.from_pretrained(bart_fp).to('cuda')


model.eval()
text = "If it's soup that you're craving, try the Yosenabe.  This is a light broth with some kind of fish (I had salmon this time), some rice noodles, veggies, oysters, clam, and shrimp.  It's my favorite!!!"
text = ' '.join(text.split()).lower()
spans = span_selector.get_spans(text)
with torch.no_grad():
    new_text = []
    new_text.append(text)
    for span in spans:
        masked_sents = text[:span['start']] + " <mask> " + text[span['end']:]
        bart_inputs = bart_tokenizer(masked_sents, return_tensors='pt').to('cuda')
        bart_outputs = bart.generate(**bart_inputs, num_beams=5,max_length=300)
        generated_text = bart_tokenizer.decode(bart_outputs[0], skip_special_tokens=True)
        new_text.append(generated_text)
    inputs = tokenizer(new_text,return_tensors='pt',padding=True,truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    print(probs[:,1])
