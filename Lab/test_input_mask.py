from transformers import BertTokenizer, BertForSequenceClassification
import torch

text = "If it's soup that you're craving, try the Yosenabe.  This is a light broth with some kind of fish (I had salmon this time), some rice noodles, veggies, oysters, clam, and shrimp.  It's my favorite!!!"
text = ' '.join(text.split()).lower()


tokenizer = BertTokenizer.from_pretrained('models/bert-base-uncased-yelp-polarity')
model = BertForSequenceClassification.from_pretrained('models/bert-base-uncased-yelp-polarity')
token = tokenizer.tokenize(text)
print(token[-8:])

inputs = tokenizer(text, return_tensors='pt')
inputs['attention_mask'] = torch.tensor((inputs['attention_mask'].shape[-1]-8)*[0]+[1]*8).unsqueeze(0)
model_outputs = model(**inputs)
print(model_outputs)
print(model_outputs.logits)
print(model_outputs.logits.softmax(dim=-1))
