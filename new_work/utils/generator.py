import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import nltk
try:
    from nltk.corpus import wordnet
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    from nltk.corpus import wordnet

class T5Generator:
    def __init__(self, model_name='models/t5-base', device='cuda'):
        self.device = device
        print(f"Loading T5 Generator: {model_name}...")
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
            self.model.eval()
        except Exception as e:
            print(f"Error loading T5: {e}")
            self.model = None

    def generate_candidates(self, text, span, num_candidates=10):
        """
        使用 T5 对 text 中的 span 进行填空
        :param text: 原始句子
        :param span: 需要替换的 span 文本
        :return: list of candidate strings
        """
        if not self.model:
            return []
            
        # 1. 构造 T5 输入: 将 span 替换为 <extra_id_0>
        # T5 的 pattern: "replace me <extra_id_0> end"
        masked_text = text.replace(span, "<extra_id_0>", 1)
        
        input_ids = self.tokenizer(masked_text, return_tensors="pt").input_ids.to(self.device)
        
        # 2. 生成
        with torch.no_grad():
            # 使用多样性采样 (Sampling) 或 Beam Search
            outputs = self.model.generate(
                input_ids, 
                num_return_sequences=num_candidates,
                do_sample=True, # 开启采样以获得多样性
                top_k=100, 
                temperature=1.2,
                top_p=0.95,
                max_length=10 # 填充的词通常不长
            )
            
        # 3. 解码与过滤
        candidates = []
        span_len = len(span.split())
        
        for output in outputs:
            # 原始解码 (保留 special tokens 以便解析哨兵)
            raw_decoded = self.tokenizer.decode(output, skip_special_tokens=False)
            
            # 清洗: T5 Infilling 输出通常是 <pad> <extra_id_0> content <extra_id_1> </s>
            # 我们需要提取 <extra_id_0> 和 <extra_id_1> 之间的内容
            decoded = raw_decoded.replace("<pad>", "").replace("</s>", "")
            
            # 提取第一个哨兵 token 后的内容
            if "<extra_id_0>" in decoded:
                parts = decoded.split("<extra_id_0>")
                if len(parts) > 1:
                    content = parts[1]
                    # 如果有下一个哨兵，截断
                    if "<extra_id_1>" in content:
                        content = content.split("<extra_id_1>")[0]
                    decoded = content.strip()
            else:
                # 如果没有哨兵，尝试直接用纯文本 (fallback)
                decoded = self.tokenizer.decode(output, skip_special_tokens=True).strip()

            # 过滤逻辑
            if not decoded: continue
            if decoded == span: continue
            
            # 长度过滤：生成的词数不能超过原 span 的 3 倍，且不能太长
            decoded_len = len(decoded.split())
            if decoded_len > span_len + 3 or decoded_len > 10:
                continue
                
            candidates.append(decoded)
        
        # --- WordNet 增强 ---
        # 如果 span 是单个词，尝试找反义词和同义词
        if span_len == 1:
            try:
                for syn in wordnet.synsets(span):
                    for lemma in syn.lemmas():
                        name = lemma.name().replace('_', ' ')
                        if name != span:
                            candidates.append(name)
                        if lemma.antonyms():
                            for ant in lemma.antonyms():
                                ant_name = ant.name().replace('_', ' ')
                                if ant_name != span:
                                    candidates.append(ant_name)
            except Exception:
                pass # WordNet 失败不影响主流程
                
        return list(set(candidates)) # 去重
