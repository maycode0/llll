import torch
import torch.nn as nn
import torch.nn.functional as F

class GatingNetwork(nn.Module):
    """
    门控网络：输入文本向量，输出每个词被选中的概率
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # 使用双向LSTM来捕捉上下文语义，决定单词重要性
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        # 输出维度为2 (选, 不选)
        self.gate_head = nn.Linear(hidden_dim * 2, 2)

    def forward(self, input_ids, tau=1.0):
        embeds = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embeds)
        logits = self.gate_head(lstm_out) # [batch, seq_len, 2]
        
        # 使用 Gumbel-Softmax 实现可微采样
        # sampling shape: [batch, seq_len, 2], 其中采样结果在最后一维
        mask_probs = F.gumbel_softmax(logits, tau=tau, hard=True)
        # 我们只需要“选中(1)”那一维的概率作为 mask
        mask = mask_probs[:, :, 1] 
        return mask

class AdvTrainer:
    def __init__(self, surrogates, gater, alpha=0.1):
        self.surrogates = surrogates # 代理模型列表 [bert, roberta, etc.]
        self.gater = gater           # 我们的门控网络
        self.alpha = alpha           # 稀疏性超参数
        self.optimizer = torch.optim.Adam(self.gater.parameters(), lr=1e-4)

    def train_step(self, input_ids, labels, mask_token_id, tau):
        self.optimizer.zero_grad()
        
        # 1. 门控网络生成 Mask
        # mask shape: [batch, seq_len], 值为 1 表示该词是“最小子集”的一员
        mask = self.gater(input_ids, tau=tau)
        
        # 2. 构造“被干扰”的输入 (用 [MASK] 替换选中的词)
        # 这里用一种软操作或直接替换：x_adv = x * (1-mask) + [MASK] * mask
        # 注意：为了保持梯度，通常在 embedding 层进行这种操作
        masked_inputs = self.apply_mask(input_ids, mask, mask_token_id)
        
        # 3. 计算跨模型的攻击损失 (Adv Loss)
        total_adv_loss = 0
        for model in self.surrogates:
            outputs = model(masked_inputs)
            # 目标：让模型对正确标签 labels 的预测概率越小越好（即 Loss 越大越好）
            # 所以我们要最小化预测正确类别的概率
            adv_loss = F.cross_entropy(outputs.logits, labels)
            total_adv_loss += adv_loss
        
        avg_adv_loss = total_adv_loss / len(self.surrogates)
        
        # 4. 计算稀疏性损失 (Sparsity Loss)
        # 目标：选中的词越少越好
        sparsity_loss = torch.mean(mask)
        
        # 5. 总目标函数：我们要最大化攻击损失 (即最小化负的攻击损失) + 最小化稀疏损失
        # 这里的目标是训练 Gater 找到那些“一旦去掉，模型就崩掉”的极少数词
        loss = -avg_adv_loss + self.alpha * sparsity_loss
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def apply_mask(self, input_ids, mask, mask_token_id):
        # 这是一个简化的演示，实际中可能需要在 Embedding 空间操作以保证梯度连续
        # 或者使用直通估计器 (Straight-Through Estimator)
        return input_ids * (1 - mask.long()) + mask_token_id * mask.long()