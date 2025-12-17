import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.processing import get_masked_sentence

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PolicyNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    def forward(self, x, action_mask):
        x = self.relu(self.fc1(x))
        # x = self.dropout(x)
        logits = self.fc2(x).squeeze(-1)# [batch ,nums_spans]
        # action masking
        HUGE_NEG = -1e9
        masked_logics = torch.where(action_mask.bool(), logits, torch.tensor(HUGE_NEG).to(logits.device))# 将掩码的内容换为一个大的数
        probs = F.softmax(masked_logics, dim=-1)
        return probs

class Agent:
    def __init__(self,input_dim, hidden_dim, output_dim, lr=2e-4, gamma=0.99, device='cuda'):
        self.device = device
        self.gamma = gamma
        self.lr = lr
        self.policy_network = PolicyNetwork(input_dim, hidden_dim, output_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=lr)
        self.log_probs = []
        self.rewards = [] 
        self.history_reward = []
    def action(self, state, action_mask,training=True):
        state = state.to(self.device)
        action_mask = action_mask.to(self.device)
        probs = self.policy_network(state, action_mask)
        if training:
            # 训练阶段：按照概率采样 (Exploration)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            # 必须保存 log_prob 用于后续反向传播
            self.log_probs.append(dist.log_prob(action))
            return action.item()
        else:
            action = torch.argmax(probs, dim=-1)
            return action.item()
    def store_reward(self, reward):
        self.rewards.append(reward)

    def clear_memory(self):
        """清空本回合数据"""
        self.log_probs = []
        self.rewards = []
    def update(self):
        """
        回合结束时调用，执行策略梯度更新 (REINFORCE)
        """
        R = 0
        policy_loss = 0
        returns = []
        
        # 1. 计算折扣回报 (Discounted Returns)
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns).to(self.device)
        
        # 2. 归一化 (稳定训练的关键)
        # if returns.std() > 0:
        if returns.std() > 1e-5:  # 只有方差足够大才归一化
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        else:
        # 方差太小，说明 Reward 几乎是一条直线，此时只减均值即可，防止除以 0 爆炸
            returns = returns - returns.mean()

        self.history_reward.append(returns.mean().item())
        # 3. 计算 Loss
        # saved_log_probs 列表里存的是 tensor，且带有梯度记录
        for log_prob, R in zip(self.log_probs, returns):
            policy_loss += (-log_prob * R)
            
        # 4. 反向传播与优化
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        # 5. 清空缓存，准备下一个 Episode
        self.clear_memory()
        return policy_loss.item()
    def clear_memory(self):
        """清空本回合数据"""
        self.log_probs = []
        self.rewards = []
    def save_model(self,path):
        torch.save(self.policy_network.state_dict(), path)
    def load_model(self,path):
        self.policy_network.load_state_dict(torch.load(path,map_location=self.device))
        self.policy_network.eval()


class AttackEnv:
    def __init__(self,local_model,local_tokenizer,global_model,global_tokenizer,fill_model,fill_tokenizer,max_steps=3):
        self.local_model = local_model
        self.local_tokenizer = local_tokenizer

        self.global_model = global_model
        self.global_tokenizer = global_tokenizer

        self.max_steps = max_steps

        self.fill_model = fill_model
        self.fill_tokenizer = fill_tokenizer

    
    def reset(self, text, true_label, spans, conflict_matrix):
        self.original_text = text
        self.current_text = text
        self.true_label = true_label

        self.spans = spans
        self.num_spans = len(self.spans)
        self.conflict_matrix = conflict_matrix
        self.visited_mask = torch.zeros(self.num_spans, dtype=torch.bool) # 记录改过谁
        self.action_mask = torch.ones(self.num_spans, dtype=torch.bool)   # 记录还能选谁
        self.steps = 0
        # 记录初始概率 (用于计算 diff reward)
        self.prev_prob, self.prev_logits = self._get_pred_prob(self.current_text)
        
        # 预计算特征 (Embedding, Scores)
        self._precompute_features()
        
        return self.get_state()
    def get_state(self):
        """
        组装状态向量: [Global, Local, Curr_Prob, Visited]
        """
        # 获取当前概率特征并广播
        curr_prob,_ = self._get_pred_prob(self.current_text)
        prob_feat = torch.tensor([curr_prob]).expand(self.num_spans, 1)
        # 拼接
        state = torch.cat([
            # self.span_embeddings,                   # 768
            self.global_scores,                     # 1
            self.local_scores,                      # 1
            prob_feat,                              # 1
            self.visited_mask.float().unsqueeze(1)  # 1
        ], dim=-1)
        # 增加 Batch 维度 [1, N, D]，同时返回 mask
        return state.unsqueeze(0), self.action_mask.unsqueeze(0)
    def step(self,action_idx,mask_token=' <mask> '):
        """
        执行动作 -> 更新环境 -> 反馈奖励
        """
        self.steps += 1
        span = self.spans[action_idx]
        # 1. 执行 BERT 填空
        # TODO: 调用 MLM 模型填空
        # new_word = self._fill_mask(self.current_text, span)
        # tokens = self.local_tokenizer.tokenize(span['text'])
        # token_lens = len(tokens)
        with torch.no_grad():
            masked_sent = self.current_text[:span['start']] + mask_token + self.current_text[span['end']:]
            fill_inputs = self.fill_tokenizer([masked_sent],return_tensors='pt', padding=True, truncation=True).to(self.fill_model.device)
            fill_outputs = self.fill_model.generate(
                **fill_inputs, 
                # do_sample=True,
                num_beams=5,
                max_length=300,
                # repetition_penalty=1.2,
                early_stopping=True,
                # no_repeat_ngram_size=2
                )
            generated_text = self.fill_tokenizer.batch_decode(fill_outputs, skip_special_tokens=True)
            # 按预测大小从小到大排序
            new_inputs = self.local_tokenizer(generated_text,return_tensors='pt', padding=True, truncation=True).to(self.local_model.device)
            new_logits = self.local_model(**new_inputs).logits
            new_probs = torch.softmax(new_logits, dim=1)[:, self.true_label]
            sorted_indices = torch.argsort(new_probs, descending=False)
            generated_text = [generated_text[i] for i in sorted_indices]
        # self.current_text = self.current_text.replace(span['text'], generated_text[0], 1)
        self.current_text = generated_text[0]
        # print(f'self.current_text:{self.current_text}')
        # 2. 更新掩码 (解决冲突)
        self.visited_mask[action_idx] = True
        # 查冲突矩阵，把冲突的也封死
        conflicts = self.conflict_matrix[action_idx]
        self.action_mask = self.action_mask & (~conflicts)
        # 3. 计算奖励
        # print(f'current_text:{self.current_text}')
        curr_prob,_ = self._get_pred_prob(self.current_text)
        # reward = (self.prev_prob - curr_prob) * 10 # 概率下降越多奖励越大
        diff_prob = self.prev_prob - curr_prob
        done = False
        reward = 0.0
        if diff_prob > 0:
            # reward -= 5
            if curr_prob < 0.5:
                reward +=(10 + diff_prob*100)
                done = True
            else:
                reward +=(1.0 + abs(diff_prob*100))
        # if curr_prob < 0.5: # 攻击成功
            # reward += 20
            # done = True
        elif diff_prob < 0:
            reward -=(1.0 + diff_prob*100)
        elif self.steps >= self.max_steps: # 步数耗尽
            reward += 0.0
            done = True
        # elif self.action_mask.sum() == 0: # 没地方可改了
        #     done = True
        self.prev_prob = curr_prob
        # print(f'curr_prob:{curr_prob}')
        return self.get_state(), reward, done

    def _get_pred_prob(self, text):
        """辅助函数：获取本地模型对正确类别的预测概率"""
        with torch.no_grad():
            inputs = self.local_tokenizer(text, return_tensors='pt').to(self.local_model.device)
            logits = self.local_model(**inputs).logits
            prob = torch.softmax(logits, dim=1)[0, self.true_label].item()
        return prob, logits[:,self.true_label].cpu()
    def _precompute_features(self):
        """辅助函数：计算所有 span 的 Score"""
        with torch.no_grad():
            spans_text = [span['text'] for span in self.spans]
            global_inputs = self.global_tokenizer(spans_text, return_tensors='pt', padding=True, truncation=True).to(self.global_model.device)
            global_outputs = self.global_model(**global_inputs)
            # 归一化
            global_logits = global_outputs.logits.cpu()
            self.global_scores = (global_logits - global_logits.min()) / (global_logits.max() - global_logits.min())
            masked_sentence = get_masked_sentence(self.original_text, self.spans, self.local_tokenizer, mask_token = '[MASK] ')
            local_inputs = self.local_tokenizer(masked_sentence.tolist(), return_tensors='pt', padding=True, truncation=True).to(self.local_model.device)
            local_outputs = self.local_model(**local_inputs)
            diff_scores = self.prev_logits - local_outputs.logits[:,self.true_label].cpu().unsqueeze(1)
            # 归一化
            self.local_scores = (diff_scores - diff_scores.min()) / (diff_scores.max() - diff_scores.min())
            
    def _fill_mask(self, text, span):
        """辅助函数：调用 BERT 填空"""
        # TODO: 这里填入你真实的填空代码
        # inputs = self.fill_tokenizer(text, return_tensors='pt')...
        # outputs = self.fill_model.generate(...)
        # new_word = self.fill_tokenizer.decode(outputs[0], skip_special_tokens=True)
        new_words = self.fill_model.generate_candidates(text, span['text'])
        return new_word