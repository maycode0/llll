from utils.rl_module import Agent, AttackEnv
from utils.processing import get_masked_sentence,get_local_important_score
from utils.processing import get_global_important_score,get_features,get_fill_spans,build_conflict_matrix,show
from transformers import BertTokenizer,BertForSequenceClassification,DistilBertTokenizer,DistilBertForSequenceClassification
from transformers import BartTokenizer, BartForConditionalGeneration
from utils.span import SpanSelector
import json
import torch
from tqdm import tqdm
import random

'''
def train_RL(datas, span_selector,Env):
    # agent = Agent()
    for date in datas:
        text = date['text']
        text = ' '.join(text.split(' ')).lower()
        # print(text)
        # break
        label = date['label']
        spans = span_selector.get_spans(text)
        conflict_matrix = build_conflict_matrix(spans)

        Env.reset(text, label, spans, conflict_matrix)

        success=False
        steps=0
        max_steps=20
        with torch.no_grad():
            text_inputs = Env.local_tokenizer(text, return_tensors='pt').to('cuda')
            local_model_outputs = Env.local_model(**text_inputs)
            probs = local_model_outputs.logits.softmax(dim=-1)[:,label]
            pred_label = probs.argmax(dim=-1).item()
            logits = local_model_outputs.logits[:,pred_label]
            print(logits)
            masked_sentence = get_masked_sentence(text,spans,Env.local_tokenizer,mask_token = '[UNK] ')
            masked_sentence_inputs = Env.local_tokenizer(masked_sentence.tolist(), return_tensors='pt', padding=True, truncation=True).to('cuda')
            local_model_outputs_mask = Env.local_model(**masked_sentence_inputs)
            logits_mask = local_model_outputs_mask.logits[:,pred_label]
            print(logits_mask)
            probs_mask = local_model_outputs_mask.logits.softmax(dim=-1)[:,label]
        local_score = (logits-logits_mask)
        print(local_score)
        break
        current_sentence = text
        while not success and steps < Env.max_steps:
            state = {
                'current_sentence':current_sentence,
                'spans':spans,
                'visited_span':visited_span
            }
            state = get_features(state)
            select = agent.action(state)

        torch.cuda.empty_cache()

    return 
'''
def train_RL(datas, span_selector,Env):
    agent = Agent(4, 128, 1)
    for data in tqdm(datas):
        text = data['text']
        text = ' '.join(text.split()).lower()
        label = data['label']
        spans = span_selector.get_spans(text)
        conflict_matrix = build_conflict_matrix(spans)
        state, action_mask = Env.reset(text, label, spans, conflict_matrix)
        down = False
        while not down:
            select = agent.action(state, action_mask)
            (next_state, new_action_mask), reward, down = Env.step(select)
            agent.store_reward(reward)
            state = next_state
            action_mask = new_action_mask
        agent.update()
        agent.clear_memory()
    show(agent.history_reward)

if __name__ == '__main__':
    # with open('datasets/yelp/yelp_70_300.json', 'r', encoding='utf-8') as f:
    with open('datasets/yelp/yelp_100_300_10k.json', 'r', encoding='utf-8') as f:
        datas = json.load(f)[:3000]
    random.shuffle(datas)
    span_selector = SpanSelector()
    local_model_fp = 'models/bert-base-uncased-yelp-polarity'
    local_model = BertForSequenceClassification.from_pretrained(local_model_fp).to('cuda')
    local_model.eval()
    local_tokenizer = BertTokenizer.from_pretrained(local_model_fp)
    global_model_fp = 'models/global_models/global_scorer_yelp'
    global_model = DistilBertForSequenceClassification.from_pretrained(global_model_fp).to('cuda')
    global_model.eval()
    global_tokenizer = DistilBertTokenizer.from_pretrained(global_model_fp)

    bart_fp = 'models/bart-base'
    bart_tokenizer = BartTokenizer.from_pretrained(bart_fp)
    fill_model = BartForConditionalGeneration.from_pretrained(bart_fp).to('cuda')
    fill_model.eval()
    Env = AttackEnv(local_model, local_tokenizer, global_model, global_tokenizer, fill_model, bart_tokenizer, max_steps=5)
    train_RL(datas, span_selector,Env)
    # train_RL(datas, span_selector,AttackEnv)

