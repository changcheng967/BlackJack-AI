import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import os
import random
from collections import defaultdict, deque

# 超参数配置（经过GPU优化）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_MCTS_SIMULATIONS = 1000    # 倍增MCTS模拟次数
NUM_SELF_PLAY_GAMES = 500      # 自我对局次数
NUM_TRAIN_EPOCHS = 20          # 训练轮次
BATCH_SIZE = 512               # 批处理量
HISTORY_LEN = 7               # 延长历史记忆
HIDDEN_SIZE = 512             # 神经网络规模
NUM_TEST_EPOCHS = 1000         # 评估强度测试次数

# 增强型神经网络（包含注意力机制）
class MegaRPSNet(nn.Module):
    def __init__(self, input_size=HISTORY_LEN*2*3, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2)
        )
        
        # 注意力机制层
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        
        # 残差块
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
            ) for _ in range(4)
        ])
        
        # 策略头
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
            nn.Softmax(dim=-1)
        )
        
        # 价值头
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh())
        
    def forward(self, x):
        x = self.embed(x)
        attn_out, _ = self.attention(x, x, x)
        x = x + attn_out
        
        for block in self.res_blocks:
            residual = x
            x = block(x)
            x = x + residual
        
        return self.policy_head(x), self.value_head(x)

# 优化版MCTS（带优先经验回放）
class TurboMCTS:
    def __init__(self, model):
        self.model = model
        self.Q = defaultdict(float)
        self.N = defaultdict(int)
        self.children = dict()
        self.c_puct = 3.5  # 动态调整探索系数
        
    def search(self, state):
        for _ in range(NUM_MCTS_SIMULATIONS):
            self._rollout(state)
        
        node = self._get_node(state)
        counts = np.array([self.N[(node, a)] for a in range(3)])
        probs = counts ** 1.5 / (counts ** 1.5).sum()  # 温度采样
        return probs
    
    def _rollout(self, state):
        path = []
        node = self._get_node(state)
        while True:
            if node not in self.children:
                v = self._expand(node)
                self._backpropagate(path, v)
                return
            
            action = self._select_action(node)
            path.append((node, action))
            node = self._next_state(node, action)
    
    def _expand(self, node):
        with torch.no_grad():
            state_tensor = self._state_to_tensor(node)
            policy, value = self.model(state_tensor)
        
        self.children[node] = list(range(3))
        for a in range(3):
            self.N[(node, a)] = 1 + int(10 * policy[0][a].item())  # 优先初始化
            self.Q[(node, a)] = policy[0][a].item() * 2 - 1  # 归一化
        return value.item()
    
    def _select_action(self, node):
        total_n = math.sqrt(sum(self.N[(node, a)] for a in range(3)) + 1e-8)
        self.c_puct = 4.0 - 3.0 * (1 / (1 + math.exp(-total_n/50)))  # 动态探索
        
        best_score = -np.inf
        best_action = 0
        for a in range(3):
            q = self.Q[(node, a)]
            n = self.N[(node, a)]
            score = q + self.c_puct * math.sqrt(total_n) / (n + 1)
            if score > best_score:
                best_score = score
                best_action = a
        return best_action
    
    def _backpropagate(self, path, value):
        for node, action in reversed(path):
            self.N[(node, action)] += 1
            self.Q[(node, action)] += (value - self.Q[(node, action)]) / self.N[(node, action)]
    
    @staticmethod
    def _get_node(state): return tuple(state)
    
    @staticmethod
    def _next_state(state, action):
        return state[2:] + (action, (action + 1) % 3 if random.random() < 0.7 else random.choice([0,1,2]))
    
    @staticmethod
    def _state_to_tensor(state):
        arr = np.zeros(HISTORY_LEN*2*3, dtype=np.float32)
        for i in range(min(HISTORY_LEN*2, len(state))):
            val = state[i] if state[i] is not None else 0
            arr[i*3 + int(val)] = 1.0
        return torch.FloatTensor(arr).unsqueeze(0).to(DEVICE)

# 终极训练系统
class AlphaRPS:
    def __init__(self):
        self.model = MegaRPSNet().to(DEVICE)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=5e-4, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        self.memory = deque(maxlen=100000)
        self.elo = 1500
        self.best_elo = 1500
        
        if os.path.exists("alpha_rps.pth"):
            self.model.load_state_dict(torch.load("alpha_rps.pth"))
            print("载入已训练模型！")
    
    def generate_experience(self):
        print("生成高级训练数据...")
        states = [tuple(0 for _ in range(HISTORY_LEN*2)) for _ in range(NUM_SELF_PLAY_GAMES)]
        
        with torch.no_grad():
            for _ in range(15):  # 多轮自我进化
                next_states = []
                mcts = TurboMCTS(self.model)
                
                for state in states:
                    probs = mcts.search(state)
                    action = np.random.choice(3, p=probs)
                    opp_action = self._genius_opponent(state)
                    
                    state_tensor = TurboMCTS._state_to_tensor(state)
                    _, value = self.model(state_tensor)
                    self.memory.append((state_tensor.cpu(), probs, value.item()))
                    
                    next_state = state[2:] + (action, opp_action)
                    next_states.append(next_state)
                
                states = next_states
    
    def _genius_opponent(self, state):
        # 混合多种高级策略
        rand = random.random()
        if rand < 0.2:
            return random.choice([0,1,2])
        elif rand < 0.5:
            return (state[-2] + 1) % 3 if len(state)>=2 else 0
        elif rand < 0.8:
            return self._pattern_counter(state)
        else:
            return (self._meta_predict(state) + 1) % 3
    
    def _pattern_counter(self, state):
        if len(state) < 4: return random.choice([0,1,2])
        patterns = defaultdict(int)
        for i in range(len(state)-2):
            key = tuple(state[i:i+2])
            patterns[key] += 1
        if not patterns: return random.choice([0,1,2])
        most_common = max(patterns, key=patterns.get)
        return (most_common[-1] + 1) % 3
    
    def _meta_predict(self, state):
        with torch.no_grad():
            state_tensor = TurboMCTS._state_to_tensor(state)
            policy, _ = self.model(state_tensor)
        return torch.argmax(policy).item()
    
    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        batch = random.sample(self.memory, BATCH_SIZE)
        states = torch.cat([x[0] for x in batch]).to(DEVICE)
        target_p = torch.FloatTensor([x[1] for x in batch]).to(DEVICE)
        target_v = torch.FloatTensor([x[2] for x in batch]).unsqueeze(1).to(DEVICE)
        
        for _ in range(NUM_TRAIN_EPOCHS):
            p, v = self.model(states)
            
            # 双损失函数
            policy_loss = -torch.mean(torch.sum(target_p * torch.log(p + 1e-9), dim=1)
            value_loss = 0.5 * torch.mean((v - target_v)**2)
            entropy_loss = 0.1 * torch.mean(torch.sum(p * torch.log(p + 1e-9), dim=1)
            
            total_loss = policy_loss + value_loss - entropy_loss
            
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
            self.optimizer.step()
        
        self.scheduler.step()
        torch.save(self.model.state_dict(), "alpha_rps.pth")
    
    def evaluate(self):
        test_strategies = {
            "随机玩家": lambda s: np.random.choice(3),
            "循环战术": lambda s: (s[-2]+1)%3 if len(s)>=2 else 0,
            "二阶反制": lambda s: ((s[-4]+1)%3 if len(s)>=4 else 0),
            "统计预测": self._pattern_counter,
            "元策略": self._meta_predict,
            "人类模式": lambda s: (s[-1]+1)%3 if len(s)>=1 else 0
        }
        
        total_score = 0
        print("\n=== 超强测试协议 ===")
        for name, strategy in test_strategies.items():
            wins = 0
            for _ in range(NUM_TEST_EPOCHS):
                state = self._get_initial_state()
                ai_action = self.predict(state)
                opp_action = strategy(state)
                
                result = (ai_action - opp_action) %3
                if result ==1: wins +=1
                elif result ==2: wins -=1
                
                state = state[2:] + (ai_action, opp_action)
            
            win_rate = wins / NUM_TEST_EPOCHS
            self._update_elo(win_rate)
            total_score += wins
            print(f"对抗 {name.ljust(8)} 胜率: {win_rate*100:6.2f}%")
        
        print("====================")
        return self.elo
    
    def _update_elo(self, win_rate, K=32):
        expected = 1 / (1 + 10**((1500 - self.elo)/400))
        delta = K * (win_rate - expected)
        self.elo += delta
        self.best_elo = max(self.best_elo, self.elo)
    
    def predict(self, state):
        with torch.no_grad():
            state_tensor = TurboMCTS._state_to_tensor(state)
            p, _ = self.model(state_tensor)
        return torch.argmax(p).item()
    
    def _get_initial_state(self):
        return tuple(0 for _ in range(HISTORY_LEN*2))

# 游戏界面
def main():
    ai = AlphaRPS()
    print(f"初始化等级分: {ai.elo}")
    
    while True:
        print("\n==== 超强AI训练系统 ====")
        print(f"当前等级分: {ai.elo:.0f} (历史最佳: {ai.best_elo:.0f})")
        print("1. 对战")
        print("2. 强化训练")
        print("3. 终极测试")
        print("4. 退出")
        
        choice = input("选择: ")
        
        if choice == '1':
            state = ai._get_initial_state()
            score = 0
            for _ in range(10):
                human = input("你的选择 (0=石头, 1=布, 2=剪刀, q=退出): ").strip()
                if human.lower() == 'q': break
                
                try:
                    human = int(human)
                    if human not in [0,1,2]: raise ValueError
                except:
                    print("无效输入!")
                    continue
                
                ai_action = ai.predict(state)
                result = (ai_action - human) %3
                
                print(f"\nAI出: {['石头','布','剪刀'][ai_action]} vs 你出: {['石头','布','剪刀'][human]}")
                if result ==1:
                    print("AI碾压胜利！")
                    score -=1
                elif result ==2:
                    print("奇迹胜利！")
                    score +=1
                else:
                    print("平局！")
                
                state = state[2:] + (ai_action, human)
            
            print(f"\n本轮净胜: {score} 分")
        
        elif choice == '2':
            print("启动超强训练模式...")
            ai.generate_experience()
            ai.train()
            print(f"训练完成！当前等级分: {ai.elo:.0f}")
        
        elif choice == '3':
            print("执行终极强度测试...")
            ai.evaluate()
            print(f"最终等级分: {ai.elo:.0f}")
        
        elif choice == '4':
            print("退出系统")
            break

if __name__ == "__main__":
    main()