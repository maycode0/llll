import numpy as np
import torch
import matplotlib.pyplot as plt

def get_spans():
    return 
def get_split_list():
    return 

def get_masked_sentence(text, spans, tokenizer, mask_token = '[MASK] '):
    masked_sentence = []
    for span in spans:
        span_text = span['text']
        token = tokenizer.tokenize(span_text)
        lens = len(token)
        start, end = span['start'], span['end']
        masked_text = text[:start] + (mask_token * lens).strip(' ') + text[end:]
        masked_sentence.append(masked_text)
    return np.array(masked_sentence)

def get_local_important_score():
    return 

def get_global_important_score():
    return 

def get_features():
    return 

def get_fill_spans():
    return 

def build_conflict_matrix(spans):
    """
    构建冲突矩阵 (N x N)
    matrix[i][j] = 1 表示 span i 和 span j 在位置上重叠
    """
    n = len(spans)
    matrix = torch.zeros((n, n), dtype=torch.bool)
    
    for i in range(n):
        for j in range(i, n):
            range_i = set(range(spans[i]['start'], spans[i]['end']))
            range_j = set(range(spans[j]['start'], spans[j]['end']))
            
            if not range_i.isdisjoint(range_j):
                matrix[i][j] = True
                matrix[j][i] = True
    return matrix

import time
def show(rewards):
    
    plt.figure(figsize=(30, 10))
    plt.plot(rewards,color = 'r',linewidth=0.4)
    plt.savefig(f'rewards_{time.time()}.png')
