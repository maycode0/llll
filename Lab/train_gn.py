import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, AdamW
import torch.nn.functional as F
from GatingNetwork import RationaleModel # 假设之前的代码保存在此文件中
import matplotlib.pyplot as plt # 导入绘图库
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
    BERT_PATH = "./your_finetuned_bert_path" # 替换为你本地微调好的BERT路径
    MAX_LEN = 128
    BATCH_SIZE = 16
    EPOCHS = 5
    LR = 1e-4
    LAMBDA_SPARSE = 0.05 # 控制子集大小，值越大，选出的词越少
    TAU_START = 1.0      # Gumbel-Softmax 起始温度
    TAU_END = 0.1        # 结束温度

    # 模拟 1000 条数据 (请替换为你的真实数据加载逻辑)
    texts = ["这是一个非常好的产品，我非常喜欢它。" for _ in range(1000)]
    labels = [1 for _ in range(1000)]

    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
    dataset = TextDataset(texts, labels, tokenizer, MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 初始化模型
    model = RationaleModel(BERT_PATH, lr=LR).to(device)

    print("开始训练 Gating Network...")
    model.train()

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
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Tau: {tau:.2f}")
        print(f"分类 Loss: {avg_adv:.4f} | 词量占比 (Sparsity): {avg_sparse:.4f}")

        # 可视化最后一条样本的选取结果
        sample_text = texts[0]
        sample_res = visualize_rationale(sample_text, mask[0], tokenizer)
        print(f"采样效果: {sample_res}\n" + "-"*30)

    # 保存训练好的 Gating Network
    torch.save(model.gating_net.state_dict(), "gating_network.pth")
    print("训练完成，Gating Network 已保存。")

if __name__ == "__main__":
    train()