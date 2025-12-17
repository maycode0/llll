'''
该文件用于获取每个文本的实体跨度
input:
    1 原始文本
    2 标签
output:
    list<Span>
'''

# span_selector.py
import spacy
import benepar
import random
from typing import List, Dict
import logging
from rich import print
logger = logging.getLogger(__name__)


class Span:
    '''
    Span:跨度实体，包含
    1 属于哪个特征句子
    2 跨度的文本是什么
    3 跨度在原始文本的开始和结束位置
    4 跨度句所属词性
    5. 跨度的长度
    6. 是否被选中
    '''
    def __init__(self, text:str, start:int, end:int, ids=-1,label=-1, pos='null',selected=False):
        self.ids = ids
        self.text = text
        self.start = start
        self.end = end
        self.label = label
        self.pos = pos
        self.len = end - start
        self.selected = selected


class SpanSelector:
    """
    混合跨度选择器：结合句法分析与滑动窗口
    """
    def __init__(self):
        try:
            self.nlp = spacy.load('en_core_web_md')
            # print('Loading en_core_web_md')
            if 'benepar' not in self.nlp.pipe_names:
                self.nlp.add_pipe('benepar', config={'model': 'benepar_en3'})
        except Exception as e:
            print(f"Warning: Benepar load failed ({e}). Falling back to heuristic mode.")
            self.nlp = spacy.load('en_core_web_sm') # Fallback

    def get_spans(self, text: str, min_span: int = 10, policy: str = 'hybrid') -> List:
        """
        提取候选跨度。
        return: [{'start': int, 'end': int, 'text': str, 'type': str}]
        """
        doc = self.nlp(text)
        spans = []

        # 策略 1: 句法跨度 (Syntactic Spans)
        if 'benepar' in self.nlp.pipe_names:
            # logger.info("Using Benepar for syntactic parsing.")
            try:
                for sent in doc.sents:
                    for const in sent._.constituents:
                        # 'NP', 'VP', 'ADJP', 'ADVP','PP', 'SBAR'且长度适中
                        # const._.labels 是一个 tuple，例如 ('NP',)
                        if const._.labels and any(label in {'NP', 'VP', 'ADJP', 'ADVP','PP', 'SBAR'} for label in const._.labels):
                            if 2 <= len(const) <= 7: 
                                spans.append(
                                    Span(
                                        text=const.text, 
                                        start=const.start_char, 
                                        end=const.end_char, 
                                        pos=const._.labels
                                        ).__dict__
                                )
            except Exception as e:
                logger.error(f"Error in Benepar parsing: {e}")
                # pass # 解析失败则跳过

        # 策略 2: 滑动窗口跨度 (Heuristic Spans)
        # 如果句法跨度太少，或者策略要求混合，则补充随机跨度
        if policy == 'heuristic' or (policy == 'hybrid' and len(spans) < min_span):
            tokens = [t for t in doc]
            # 随机生成几个长度为 2-6 的跨度
            for _ in range(5):
                span_len = random.randint(2, 6)
                if len(tokens) - span_len > 0:
                    start_idx = random.randint(0, len(tokens) - span_len)
                    end_idx = start_idx + span_len
                    # 转换为字符索引
                    span_obj = doc[start_idx:end_idx]
                    spans.append(
                            Span(
                                text=span_obj.text, 
                                start=span_obj.start_char, 
                                end=span_obj.end_char, 
                                pos=('RANDOM', )
                            ).__dict__
                    )

        # 去重逻辑就是使用 start 和 end 作为唯一键
        unique_spans = []
        seen = set()
        for s in spans:
            key = (s['start'], s['end'])
            if key not in seen:
                seen.add(key)
                unique_spans.append(s)
        
        return unique_spans