import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np

class GatingNetwork(nn.Module):
    def __init__(self, bert_hidden_dim=768, rnn_hidden_dim=256):
        super(GatingNetwork, self).__init__()
        self.rnn = nn.GRU(bert_hidden_dim, rnn_hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(rnn_hidden_dim * 2, 2) # 输出 [Mask, Keep] 的 Logits

    def forward(self, x_embeddings, tau=1.0, hard=True):
        # x_embeddings: [batch, seq_len, 768]
        rnn_out, _ = self.rnn(x_embeddings)
        logits = self.fc(rnn_out) # [batch, seq_len, 2]
        
        # Gumbel-Softmax 采样
        # z: [batch, seq_len, 2], 其中 z[:,:,1] 是 Keep 的概率/掩码
        z = F.gumbel_softmax(logits, tau=tau, hard=hard)
        # 我们只需要“选中(1)”那一维的概率作为 mask
        mask = z[:, :, 1]
        return mask, logits

class SCMClassifier(nn.Module):
    def __init__(self, model_name='models/bert-base-uncased-yelp-polarity', num_labels=2):
        super(SCMClassifier, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.generator = GatingNetwork()
        
        # 冻结 BERT 参数，只训练生成器（这是解释现有模型的标准做法）
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask, tau=1.0, hard=True):
        # 第一路：获取上下文 Embedding (不计算梯度以提速)
        with torch.no_grad():
            outputs = self.bert.bert(input_ids, attention_mask=attention_mask)
            context_embeddings = outputs.last_hidden_state
            raw_embeddings = self.bert.bert.embeddings(input_ids)

        # 第二路：生成 Mask
        mask, _ = self.generator(context_embeddings, tau=tau, hard=hard)
        
        # 强制保留 [CLS] 位 (index 0)
        mask_with_cls = mask.clone()
        mask_with_cls[:, 0] = 1.0
        
        # 第三路：应用 Mask 并重新分类
        masked_embeddings = raw_embeddings * mask_with_cls.unsqueeze(-1)
        # 注意：这里需要传入 inputs_embeds
        logits = self.bert(inputs_embeds=masked_embeddings, attention_mask=attention_mask).logits
        
        return logits, mask

def scm_loss_fn(logits, labels, mask, attention_mask, lambda_sparse, gamma_cont):
    # 1. 分类准确性 (Cross Entropy)
    loss_acc = F.cross_entropy(logits, labels)
    
    # 2. 稀疏性惩罚 (L0 近似)
    # 只统计 attention_mask 为 1 的有效 token
    active_tokens = attention_mask.sum(dim=1)
    loss_sparse = (mask * attention_mask).sum(dim=1) / active_tokens
    loss_sparse = loss_sparse.mean()
    
    # 3. 连贯性惩罚 (Continuity)
    diff = torch.abs(mask[:, 1:] - mask[:, :-1])
    loss_cont = (diff * attention_mask[:, 1:]).sum(dim=1) / active_tokens
    loss_cont = loss_cont.mean()
    
    total_loss = loss_acc + lambda_sparse * loss_sparse + gamma_cont * loss_cont
    return total_loss, loss_acc, loss_sparse, loss_cont


def get_full_word_rationale(tokenizer, input_ids, mask):
    """
    将子词级别的 mask 映射回完整的单词，并保持逻辑连贯。
    """
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    full_words = []
    current_word = ""
    is_current_word_selected = False
    
    # 特殊符号列表
    special_tokens = [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]

    for i, token in enumerate(tokens):
        # 跳过特殊符号
        if token in special_tokens:
            continue
            
        # 判断是否是子词（以 ## 开头）
        if token.startswith("##"):
            # 继续拼接当前单词
            current_word += token[2:]
            # 只要子词序列中有一个被选中，整个词就视为选中
            if mask[i] > 0.5:
                is_current_word_selected = True
        else:
            # 在开始新单词前，先保存上一个单词（如果存在）
            if current_word:
                full_words.append((current_word, is_current_word_selected))
            
            # 开始处理新单词
            current_word = token
            is_current_word_selected = (mask[i] > 0.5)

    # 循环结束后，别忘了添加最后一个单词
    if current_word:
        full_words.append((current_word, is_current_word_selected))

    # 过滤出被选中的完整单词
    selected_words = [word for word, selected in full_words if selected]
    
    # 还原为带空格的句子（针对英文）
    rationale_text = " ".join(selected_words)
    
    return rationale_text, full_words

def generate_masked_sentence(all_word_stats, mask_token="<mask>"):
    """
    根据映射统计结果，将非核心词替换为指定的 mask 占位符。
    """
    final_tokens = []
    
    for word, is_selected in all_word_stats:
        if is_selected:
            # 如果是核心关键词，保持原样
            final_tokens.append(word)
        else:
            # 如果是非核心词，替换为 mask 占位符
            final_tokens.append(mask_token)
            
    # 将单词重新拼接回句子
    masked_sentence = " ".join(final_tokens)
    return masked_sentence

def run_scm_demo():
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 初始化
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = SCMClassifier().to(device)
    optimizer = torch.optim.Adam(model.generator.parameters(), lr=1e-4)

    # 模拟数据
    text = "If it's soup that you're craving, try the Yosenabe.  This is a light broth with some kind of fish (I had salmon this time), some rice noodles, veggies, oysters, clam, and shrimp.  It's my favorite!!!"
    texts = [' '.join(text.split()).lower()]
    labels = torch.tensor([1]).to(device) # 1: 正面, 0: 负面
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)

    # 超参数
    lambda_sparse = 0.07  # 稀疏系数，越大则选出的词越少
    gamma_cont = 0.05    # 连贯系数，越大则选出的词越趋向于连续短语
    
    # --- 训练阶段 ---
    model.train()
    print("\n--- Starting Training ---")
    for epoch in range(100): # 简化演示，训练 50 次
        # 温度退火
        tau = max(0.1, 1.0 * np.exp(-0.01 * epoch))
        
        logits, mask = model(inputs['input_ids'], inputs['attention_mask'], tau=tau)
        loss, l_acc, l_spa, l_con = scm_loss_fn(logits, labels, mask, inputs['attention_mask'], lambda_sparse, gamma_cont)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:02d} | Loss: {loss.item():.4f} | AccLoss: {l_acc:.4f} | Sparsity: {l_spa:.4f}")

    # --- 推理阶段 (提取最小子集) ---
    model.eval()
    print("\n--- Starting Inference (Rationale Extraction) ---")
    with torch.no_grad():
        # 推理时 hard=True 直接获得 0/1 掩码
        _, final_masks = model(inputs['input_ids'], inputs['attention_mask'], tau=0.1, hard=True)
        
        for i, text in enumerate(texts):
            input_id_list = inputs['input_ids'][i].tolist()
            mask_list = final_masks[i].tolist()
            # 使用全词映射函数
            rationale_text, all_word_stats = get_full_word_rationale(tokenizer, input_id_list, mask_list)
            print(f"\nOriginal: {text}")
            print(f"Minimal Sufficient Subset: {rationale_text}")
            # 进阶：如果你想看哪些词被丢弃了，可以打印 all_word_stats
            masked_ver = generate_masked_sentence(all_word_stats, mask_token="<mask>")
            print(f"Masked Version: {masked_ver}")
# if __name__ == "__main__":
#     run_scm_demo()


