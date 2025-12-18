import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification, BertModel,AdamW
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
import matplotlib.pyplot as plt # 导入绘图库

class GatingNetwork(nn.Module):
    def __init__(self, bert_hidden_dim=768, rnn_hidden_dim=256):
        super(GatingNetwork, self).__init__()
        self.rnn = nn.GRU(bert_hidden_dim, rnn_hidden_dim, batch_first=True, bidirectional=True) # 这里双向GRU捕捉上下文
        self.fc = nn.Linear(rnn_hidden_dim * 2, 2)  # 输出 2 维，对应 Keep 和 Drop

    def forward(self, bert_embeddings,tau=1.0, hard = True): # tau越小越接近one-hot
        # bert_embeddings: [batch, seq_len, bert_hidden_dim]
        rnn_out, _ = self.rnn(bert_embeddings) # rnn_out: [batch, seq_len, rnn_hidden_dim * 2]
        logits = self.fc(rnn_out) # logits: [batch, seq_len, 2]
        
        # Gumbel-Softmax 采样
        # z: [batch, seq_len, 2], 其中 z[:,:,1] 是 Keep 的概率/掩码
        z = F.gumbel_softmax(logits, tau=tau, hard=hard)
        # 我们只需要“选中(1)”那一维的概率作为 mask
        mask = z[:, :, 1]
        return mask, logits

class RationaleModel(nn.Module):
    def __init__(self, bert_model_path, lr=1e-4, models_list=[]):
        super(RationaleModel, self).__init__()
        # 1. 加载你本地微调好的 BERT
        self.bert = BertForSequenceClassification.from_pretrained(bert_model_path)
        self.bert.eval()
        # 冻结 BERT 参数，因为我们只需要训练 GatingNetwork
        for param in self.bert.parameters():
            param.requires_grad = False
        self.models = models_list
        # 2. 初始化 Gating Network
        # embed_dim 通常是 768 (BERT-base)
        self.gating_net = GatingNetwork(768, 256)
        self.optimizer = torch.optim.Adam(self.gating_net.parameters(), lr=lr)

    def forward(self, input_ids, attention_mask,tau=1.0):
        # A. 获取 BERT 的初始 Embedding (未经过 Transformer 层)
        # 这是为了给 Gating Network 提供输入特征
        with torch.no_grad(): # 获取具有上下文的embedding
            outputs = self.bert.bert(input_ids, attention_mask=attention_mask)
            last_hidden = outputs.last_hidden_state #经过Transformer层后的信息，包含[CLS]
            raw_embeddings = self.bert.bert.embeddings(input_ids) # [batch, seq_len, 768]，未经过Transformer层
        mask, _ = self.gating_net(last_hidden,tau=tau)

        # 强制保留 [CLS] 位 (index 0)
        # mask[:, 0] = 1.0 #[1,seq_len]
        # C. 【优化】强制保留 [CLS] 位，且不破坏梯度
        cls_mask = torch.ones((mask.shape[0], 1), device=mask.device)
        mask = torch.cat([cls_mask, mask[:, 1:]], dim=1)

        # 应用 Mask 并重新分类
        masked_embeddings = raw_embeddings * mask.unsqueeze(-1) # [1,seq_len,1] mask对齐
        # 注意：这里需要传入 inputs_embeds
        logits = self.bert(inputs_embeds=masked_embeddings, attention_mask=attention_mask).logits# 计算未经过Transformer层的logits
        return logits, mask

    def train_step(self, logits, labels, mask, attention_mask, lambda_sparse=0.1):
        self.optimizer.zero_grad()
        adv_loss = 0
        # 1. 分类准确性 (Cross Entropy),越小越好
        adv_loss = F.cross_entropy(logits, labels)
        # 2. 稀疏性惩罚 (L0 近似),越大则选出的词越少
        active_tokens = attention_mask.sum(dim=1)
        sparsity_loss = (mask * attention_mask).sum(dim=1) / active_tokens
        sparsity_loss = sparsity_loss.mean()
        total_loss = adv_loss + lambda_sparse * sparsity_loss
        total_loss.backward()
        self.optimizer.step()
        return adv_loss.item(), sparsity_loss.item(), total_loss.item()

#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
# --- 1. 数据准备 ---
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        encoding = self.tokenizer(
            self.texts[item],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(self.labels[item], dtype=torch.long)
        }
# --- 2. 可视化工具：查看模型选了哪些词 ---
def visualize_rationale(text, mask, tokenizer):
    """
    将 mask 作用于文本并打印，被选中的词会加上中括号
    """
    tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text, truncation=True, max_length=128))
    # mask 形状为 [seq_len], 转换为 numpy
    m = mask.cpu().detach().numpy()
    
    result = []
    for i, token in enumerate(tokens):
        if i >= len(m): break
        if token in ['[PAD]', '[CLS]', '[SEP]']: continue
        
        if m[i] > 0.5: # 被选中的词
            result.append(f"【{token}】")
        else:
            result.append(token)
    return "".join(result).replace("##", "")

# --- 3. 主训练流程 ---
def train():
    # 超参数设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BERT_PATH = "models/bert-base-uncased-yelp-polarity" # 替换为你本地微调好的BERT路径
    MAX_LEN = 512
    BATCH_SIZE = 16
    EPOCHS = 5
    LR = 1e-4
    LAMBDA_SPARSE = 0.05 # 控制子集大小，值越大，选出的词越少
    TAU_START = 1.0      # Gumbel-Softmax 起始温度
    TAU_END = 0.1        # 结束温度

    # 模拟 1000 条数据 (请替换为你的真实数据加载逻辑)
    # texts = ["这是一个非常好的产品，我非常喜欢它。" for _ in range(1000)]
    # labels = [1 for _ in range(1000)]
    # 加载 Yelp 数据集
    start = 0
    end = 1000
    with open('datasets/yelp/yelp_100_300_10k.json', 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    texts = [' '.join(item['text'].split()).lower() for item in train_data[start:end]]
    labels = [item['label'] for item in train_data[start:end]]

    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
    dataset = TextDataset(texts, labels, tokenizer, MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 初始化模型
    model = RationaleModel(BERT_PATH, lr=LR).to(device)

    print("开始训练 Gating Network...")
    model.train()
    # --- 用于记录损失的列表 ---
    history = {
        'adv_loss': [],
        'sparsity_loss': [],
        'total_loss': []
    }
    for epoch in range(EPOCHS):
        # 计算当前 Epoch 的温度 (线性退火)
        tau = TAU_START - (TAU_START - TAU_END) * (epoch / EPOCHS)
        
        total_adv_loss = 0
        total_sparse_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=True)
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # 前向传播
            logits, mask = model(input_ids, attention_mask, tau=tau)
            
            # 训练步：计算 Loss 并更新
            adv_loss, sparse_loss, total_loss = model.train_step(
                logits, labels, mask, attention_mask, lambda_sparse=LAMBDA_SPARSE
            )

            total_adv_loss += adv_loss
            total_sparse_loss += sparse_loss
            # --- 实时更新进度条右侧的统计信息 ---
            pbar.set_postfix({
                'adv_loss': f"{adv_loss:.4f}",
                'sparsity': f"{sparse_loss:.4f}",
                'tau': f"{tau:.2f}"
            })


        # 每个 Epoch 结束打印状态
        avg_adv = total_adv_loss / len(dataloader)
        avg_sparse = total_sparse_loss / len(dataloader)
        avg_total = epoch_total / len(dataloader)

        history['adv_loss'].append(avg_adv)
        history['sparsity_loss'].append(avg_sparse)
        history['total_loss'].append(avg_total)

        print(f"Epoch {epoch+1}/{EPOCHS} | Tau: {tau:.2f}")
        print(f"分类 Loss: {avg_adv:.4f} | 词量占比 (Sparsity): {avg_sparse:.4f}")

        # 可视化最后一条样本的选取结果
        sample_text = texts[0]
        sample_res = visualize_rationale(sample_text, mask[0], tokenizer)
        print(f"采样效果: {sample_res}\n" + "-"*30)


    # --- 4. 绘制并保存折线图 ---
    plt.figure(figsize=(10, 6))
    epochs_range = range(1, EPOCHS + 1)

    plt.plot(epochs_range, history['adv_loss'], label='Classification Loss (Adv)', marker='o')
    plt.plot(epochs_range, history['sparsity_loss'], label='Sparsity Loss', marker='s')
    plt.plot(epochs_range, history['total_loss'], label='Total Loss', linestyle='--', marker='^')

    plt.title('Training Losses of Gating Network')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True)

    # 保存训练好的 Gating Network
    torch.save(model.gating_net.state_dict(), "gating_network.pth")
    print("训练完成，Gating Network 已保存。")

train()