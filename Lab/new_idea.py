import torch
import torch.nn as nn
import torch.nn.functional as F

class SCMMaskGenerator(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        # 一个简单的双向 GRU 或 Transformer 都可以作为生成器
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        # 输出维度为 2，分别代表该 token 被预测为 0 (Mask) 或 1 (Keep) 的 Logits
        self.fc = nn.Linear(hidden_dim * 2, 2)

    def forward(self, x_embeddings, temperature=1.0):
        # x_embeddings: [batch_size, seq_len, embed_dim]
        h, _ = self.rnn(x_embeddings)
        logits = self.fc(h) # [batch_size, seq_len, 2]
        
        # 使用 Gumbel-Softmax 进行重参数化采样
        # hard=True 会在梯度传播时使用连续值，但在前向计算时强制转为 0 或 1
        z_sampled = F.gumbel_softmax(logits, tau=temperature, hard=True)
        
        # 我们只需要“保留(1)”那一侧的概率/掩码
        mask = z_sampled[:, :, 1] 
        return mask, logits[:, :, 1]




def scm_loss_function(pred_logits, target, mask, lambda_sparse, gamma_cont):
    """
    pred_logits: 经过 mask 后分类器的输出
    target: 真实标签
    mask: 生成器产生的 0/1 向量 [batch, seq_len]
    """
    # 1. 准确性损失 (Accuracy Loss)
    loss_acc = F.cross_entropy(pred_logits, target)
    
    # 2. 稀疏性损失 (Sparsity Loss: L0 范数的近似)
    # 我们希望 mask 中 1 的数量尽可能少
    loss_sparse = torch.mean(torch.sum(mask, dim=1))
    
    # 3. 连贯性损失 (Continuity Penalty)
    # 计算相邻 mask 值的差值，差值越多说明越破碎
    # loss = sum(|z_i - z_{i-1}|)
    loss_cont = torch.mean(torch.sum(torch.abs(mask[:, 1:] - mask[:, :-1]), dim=1))
    
    # 总损失
    total_loss = loss_acc + lambda_sparse * loss_sparse + gamma_cont * loss_cont
    
    return total_loss, loss_acc, loss_sparse, loss_cont

# --- 模拟训练流程 ---
from transformers import BertTokenizer, BertForSequenceClassification
# 假设已经有一个训练好的分类器模型 classifier
# classifier.eval() # 解释阶段通常冻结分类器
# for param in classifier.parameters():
#     param.requires_grad = False
text = "If it's soup that you're craving, try the Yosenabe.  This is a light broth with some kind of fish (I had salmon this time), some rice noodles, veggies, oysters, clam, and shrimp.  It's my favorite!!!"
text = ' '.join(text.split()).lower()

tokenizer = BertTokenizer.from_pretrained('models/bert-base-uncased-yelp-polarity')
model = BertForSequenceClassification.from_pretrained('models/bert-base-uncased-yelp-polarity')
model.eval()
# 获取有上下文的embedding
with torch.no_grad():
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model.bert(**inputs)
    last_hidden_state = outputs.last_hidden_state
probs = model(**inputs).logits.softmax(dim=-1)

# print(last_hidden_state.shape)
generator = SCMMaskGenerator(embed_dim=768, hidden_dim=256)
# optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)
mask, mask_logits = generator(last_hidden_state)
mask[:, 0] = 1.0
mask = mask.unsqueeze(-1)
print(mask)
# print(mask.shape)
masked_embeddings = last_hidden_state * mask
print(masked_embeddings)
masked_outputs = model(inputs_embeds=masked_embeddings)
logits = masked_outputs.logits
print(logits.softmax(dim=-1))
print(probs)
# print(mask_logits)
# x_masked = last_hidden_state * mask.unsqueeze(-1)
# print(x_masked[:,0,:])
# 在训练循环中：
# 1. 获取输入 x 的 embedding
# 2. mask, mask_logits = generator(x_embeddings, temperature=0.5)
# 3. x_masked = x_embeddings * mask.unsqueeze(-1)
# 4. out = classifier(inputs_embeds=x_masked)
# 5. loss, _, _, _ = scm_loss_function(out, y, mask, lambda_sparse=0.01, gamma_cont=0.1)
# 6. loss.backward()
# 7. optimizer.step()
