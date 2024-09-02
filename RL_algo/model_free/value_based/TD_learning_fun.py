"""
    TD learning based on function 算法
    

    代码库中包括 DQN , CNN DQN , Double DQN , Dueling DQN 

    Author: Z.R.Wang
    Aug,26,2024 in YSU
    Emile: wangzhanran@stumail.ysu.edu.cn

    reference:强化学习的数学原理(西湖大学赵世钰老师),动手学强化学习https://github.com/boyu-ai/Hands-on-RL
"""

import random
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import RL_algo.rl_utils as rl_utils
import gym

# 离散动作转回连续的函数
def dis_to_con(discrete_action, env, action_dim):  
    action_lowbound = env.action_space.low[0]  # 连续动作的最小值
    action_upbound = env.action_space.high[0]  # 连续动作的最大值
    re = action_lowbound + (discrete_action /(action_dim - 1)) * (action_upbound -action_lowbound)
    # print(re)
    return re

class ReplayBuffer:
    '''
        经验回放池
        capacity: int, 经验回放池的容量
    '''
    def __init__(self,capacity) -> None:
        '''
            collections.deque(maxlen=capacity) 是 Python 中 collections 模块中的 deque 类的一个实例化方法。
            deque 是双端队列（double-ended queue）的缩写，它允许在两端高效地添加和删除元素。
            以下是 collections.deque(maxlen=capacity) 的详细解释：
            collections.deque：从 collections 模块导入 deque 类。
            maxlen=capacity：指定双端队列的最大长度为 capacity。
                            如果 maxlen 被设置，当队列达到最大长度时，再添加新元素会自动移除最旧的元素
        '''
        self.buffer = collections.deque(maxlen=capacity) # 队列,先进先出

    def add(self, state, action, reward, next_state, done):
        '''
            添加经验到经验回放池
            state: np.ndarray, 当前状态
            action: int, 当前动作
            reward: float, 当前奖励
            next_state: np.ndarray, 下一个状态
            done: bool, 是否结束
        '''
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        '''
            从经验回放池中采样数据
            batch_size: int, 批次大小
        '''
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done
    
    def size(self):
        '''
            目前buffer中数据的数量
        '''
        return len(self.buffer)
    

class Qnet(torch.nn.Module):
    '''
        多层隐藏层的Q网络
        state_dim: int, 状态空间的维度
        action_dim: int, 动作空间的维度
        hidden_dims: list of int, 每个隐藏层的维度
        示例用法
        state_dim = 4
        action_dim = 2
        hidden_dims = [128, 64, 32] 三个隐藏层，维度分别为128, 64, 32
    '''
    def __init__(self, state_dim, action_dim, hidden_dims) -> None:
        super(Qnet, self).__init__()
        self.layers = torch.nn.ModuleList()
        
        # 输入层到第一个隐藏层
        self.layers.append(torch.nn.Linear(state_dim, hidden_dims[0]))
        
        # 隐藏层之间的连接
        for i in range(1, len(hidden_dims)):
            self.layers.append(torch.nn.Linear(hidden_dims[i-1], hidden_dims[i]))
        
        # 最后一个隐藏层到输出层
        self.layers.append(torch.nn.Linear(hidden_dims[-1], action_dim))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))  # 隐藏层使用ReLU激活函数
        return self.layers[-1](x)  # 输出层不使用激活函数

# class Qnet(torch.nn.Module):
#     ''' 只有一层隐藏层的Q网络 '''
#     def __init__(self, state_dim, action_dim, hidden_dim) -> None:
#         super(Qnet, self).__init__()
#         self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
#         self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
#         return self.fc2(x)


class ConvolutionalQnet(torch.nn.Module):
    ''' 加入卷积层的Q网络 '''
    def __init__(self, action_dim, in_channels=4):
        super(ConvolutionalQnet, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = torch.nn.Linear(7 * 7 * 64, 512)
        self.head = torch.nn.Linear(512, action_dim)

    def forward(self, x):
        x = x / 255
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # 展平操作
        x = F.relu(self.fc4(x))
        return self.head(x)
    

class VAnet(torch.nn.Module):
    ''' 只有一层隐藏层的A网络和V网络 '''
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(VAnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)  # 共享网络部分
        self.fc_A = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_V = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        A = self.fc_A(F.relu(self.fc1(x)))
        V = self.fc_V(F.relu(self.fc1(x)))
        Q = V + A - A.mean(1).view(-1, 1)  # Q值由V值和A值计算得到
        return Q


class DQN:
    '''
        DQN算法
        state_dim: int, 状态空间的维度
        action_dim: int, 动作空间的维度
        hidden_dim: int, 隐藏层的维度
        learning_rate: float, 学习率
        gamma: float, 折扣因子
        epsilon: float, ε-greedy策略中的ε
        target_update: int, 目标网络更新频率
        device: torch.device, 设备(cpu或gpu)
    '''
    def __init__(self,state_dim, action_dim, hidden_dims, learning_rate, 
                 gamma, epsilon, target_update, device,env) -> None:
        self.action_dim = action_dim
        self.main_q_net = Qnet(state_dim, action_dim, hidden_dims).to(device) # Q网络，也就是我们要训练的网络
        self.target_q_net = Qnet(state_dim, action_dim, hidden_dims).to(device) # 目标网络,就是隔一段时间更新一次
        self.optimizer = torch.optim.Adam(self.main_q_net.parameters(), lr=learning_rate) #使用Adam优化器
        self.gamma = gamma # 折扣因子
        self.epsilon = epsilon # ε-greedy策略中的ε
        self.target_update = target_update # 目标网络更新频率
        self.count = 0 # 计数器,记录更新次数
        self.device = device
        self.env = env
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            # print("动作空间是离散的")
            self.action_type = 'discrete'
        elif isinstance(self.env.action_space, gym.spaces.Box):
            # print("动作空间是连续的")
            self.action_type = 'continuous'
        else:
            # print("动作空间是其他类型")
            raise ValueError("Unsupported action space type")

    def take_action(self, state):
        '''
            ε-greedy策略选择动作
            state: np.ndarray, 当前状态
        '''
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.main_q_net(state).argmax().item()
        return action

    def best_action(self, state):
        '''
            选择最优动作
            state: np.ndarray, 当前状态
        '''
        state = torch.FloatTensor(state).to(self.device)# 将state转换为张量
        q_value = self.main_q_net(state)# 计算Q值
        action = torch.argmax(q_value).item()# 选择Q值最大的动作
        if self.action_type == 'continuous':
            action = [dis_to_con(action, self.env, self.action_dim)]
        return action
        
    def update(self,transition_dict):
        '''
            更新Q网络
            transition_dict: dict, 从经验回放池中采样的数据
        '''
        # 将数据转换为张量
        states = torch.FloatTensor(transition_dict['states']).to(self.device)
        actions = torch.LongTensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.FloatTensor(transition_dict['rewards']).view(-1, 1).to(self.device)
        next_states = torch.FloatTensor(transition_dict['next_states']).to(self.device)
        dones = torch.FloatTensor(transition_dict['dones']).view(-1, 1).to(self.device)

        # 计算Q值和目标Q值
        q_values = self.main_q_net(states).gather(1, actions)  # Q值，Q(s,a)
        # DQN
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)# 下个状态的最大Q值
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD误差目标, TD_target = r + γmaxQ(s',a')

        # 更新Q网络
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数,TD误差TD_error = Q_target - Q(s,a)
        self.optimizer.zero_grad() # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward() # 反向传播更新参数
        self.optimizer.step() # 更新参数

        # 每隔一段时间更新目标网络
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.main_q_net.state_dict()) # 更新目标网络
        self.count += 1


    def train_DQN(self,buffer_size,num_episodes,minimal_size,batch_size):
        """ 
            运行DQN算法 
            buffer_size: int, 经验回放池的容量
            num_episodes: int, 训练的序列数量
            minimal_size: int, 经验回放池中数据的最小数量
            batch_size: int, 批次大小
        """
        replay_buffer = ReplayBuffer(buffer_size) # 经验回放池
        return_list = []  # 记录每一条序列的回报
        episodes_len_list = [] # 记录每一条序列的长度
        for i in range(10): #显示10个进度条
            # tqdm的进度条功能
            with tqdm(total=int(num_episodes/10) , desc='Iteration %d' % i) as pbar:
                for i_episode in range(int(num_episodes/10)):
                    st = self.env.reset()
                    done = False
                    episode_return = 0
                    episodes_len = 0
                    while not done:
                        at = self.take_action(st)
                        # print(at)
                        if self.action_type == 'continuous':
                            at_con = [dis_to_con(at, self.env, self.action_dim)]
                            st1,rt1,down_,_ = self.env.step(at_con)
                        else:
                            st1,rt1,down_,_ = self.env.step(at)
                        replay_buffer.add(st, at, rt1, st1, down_)
                        episode_return += rt1
                        st = st1
                        done = down_
                        episodes_len += 1
                        # 当buffer数据的数量超过一定值后,才进行Q网络训练
                        if replay_buffer.size() > minimal_size:
                            b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                            transition_dict = {
                                'states': b_s,
                                'actions': b_a,
                                'rewards': b_r,
                                'next_states': b_ns,
                                'dones': b_d
                            }
                            self.update(transition_dict)
                    return_list.append(episode_return)
                    episodes_len_list.append(episodes_len)
                    if (i_episode + 1) % 10 == 0:# 每10条序列打印一下这10条序列的平均回报
                        pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                        })
                    pbar.update(1)
        return return_list,episodes_len_list

    
    def plot_DQN(self,return_list,episodes_len_list):
        """ 绘制 DQN算法的学习曲线 """
        episodes_list = list(range(len(return_list)))
        plt.figure(figsize=(10, 8))
        plt.suptitle('DQN on {}'.format(self.env.spec.id))

        plt.subplot(3, 1, 1)
        plt.plot(episodes_list, return_list)
        plt.ylabel('Total rewards')

        mv_return = rl_utils.moving_average(return_list, 9)
        plt.subplot(3, 1, 2)
        plt.plot(episodes_list, mv_return)
        plt.ylabel('Moving average rewards')

        plt.subplot(3, 1, 3)
        plt.plot(episodes_list, episodes_len_list)
        plt.xlabel('Episodes index')
        plt.ylabel('Episodes length')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    def run_DQN(self,buffer_size,num_episodes,minimal_size,batch_size):
        
        ''' 
            运行DQN算法
            buffer_size: int, 经验回放池的容量
            num_episodes: int, 训练的序列数量
            minimal_size: int, 经验回放池中数据的最小数量
            batch_size: int, 批次大小
        '''
        return_list,episodes_len_list = self.train_DQN(buffer_size,num_episodes,minimal_size,batch_size)
        self.plot_DQN(return_list,episodes_len_list)

## 没有测试
class CNN_DQN:
    '''
        CNN DQN算法
        action_dim: int, 动作空间的维度
        learning_rate: float, 学习率
        gamma: float, 折扣因子
        epsilon: float, ε-greedy策略中的ε
        target_update: int, 目标网络更新频率
        device: torch.device, 设备(cpu或gpu)
    '''
    def __init__(self, action_dim, learning_rate, gamma, epsilon, target_update, device, env):
        self.action_dim = action_dim
        self.main_q_net = ConvolutionalQnet(action_dim).to(device)
        self.target_q_net = ConvolutionalQnet(action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.main_q_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
        self.device = device
        self.env = env
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            # print("动作空间是离散的")
            self.action_type = 'discrete'
        elif isinstance(self.env.action_space, gym.spaces.Box):
            # print("动作空间是连续的")
            self.action_type = 'continuous'
        else:
            # print("动作空间是其他类型")
            self.action_type = 'other'

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_dim)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_value = self.main_q_net(state)
            action = torch.argmax(q_value).item()
            return action

    def update(self, transition_dict):
        states = torch.FloatTensor(transition_dict['states']).to(self.device)
        actions = torch.LongTensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.FloatTensor(transition_dict['rewards']).view(-1, 1).to(self.device)
        next_states = torch.FloatTensor(transition_dict['next_states']).to(self.device)
        dones = torch.FloatTensor(transition_dict['dones']).view(-1, 1).to(self.device)

        q_values = self.main_q_net(states).gather(1, actions)
        max_next_q = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q * (1 - dones)

        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.main_q_net.state_dict())
        self.count += 1

    def train_CNN_DQN(self, buffer_size, num_episodes, minimal_size, batch_size):
        replay_buffer = ReplayBuffer(buffer_size)
        return_list = []
        episodes_len_list = []
        for i in range(10):
            with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
                for i_episode in range(int(num_episodes / 10)):
                    state = self.env.reset()
                    state = self.env.render(mode='rgb_array').transpose((2, 0, 1))
                    done = False
                    episode_return = 0
                    episodes_len = 0
                    while not done:
                        if self.action_type == 'discrete':
                            action = self.take_action(state)
                        if self.action_type == 'continuous':
                            action = [dis_to_con(action, self.env, self.action_dim)]
                        next_state, reward, done, _ = self.env.step(action)
                        next_state = self.env.render(mode='rgb_array').transpose((2, 0, 1))
                        replay_buffer.add(state, action, reward, next_state, done)
                        episode_return += reward
                        state = next_state
                        episodes_len += 1
                        if replay_buffer.size() > minimal_size:
                            b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                            transition_dict = {
                                'states': b_s,
                                'actions': b_a,
                                'rewards': b_r,
                                'next_states': b_ns,
                                'dones': b_d
                            }
                            self.update(transition_dict)
                    return_list.append(episode_return)
                    episodes_len_list.append(episodes_len)
                    if (i_episode + 1) % 10 == 0:
                        pbar.set_postfix({
                            'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                            'return': '%.3f' % np.mean(return_list[-10:])
                        })
                    pbar.update(1)
        return return_list, episodes_len_list

    def plot_CNN_DQN(self, return_list, episodes_len_list):
        episodes_list = list(range(len(return_list)))
        plt.figure(figsize=(10, 8))
        plt.suptitle('CNN-DQN on {}'.format(self.env.spec.id))

        plt.subplot(3, 1, 1)
        plt.plot(episodes_list, return_list)
        plt.ylabel('Total rewards')

        mv_return = rl_utils.moving_average(return_list, 9)
        plt.subplot(3, 1, 2)
        plt.plot(episodes_list, mv_return)
        plt.ylabel('Moving average rewards')

        plt.subplot(3, 1, 3)
        plt.plot(episodes_list, episodes_len_list)
        plt.xlabel('Episodes index')
        plt.ylabel('Episodes length')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    def run_CNN_DQN(self, buffer_size, num_episodes, minimal_size, batch_size):
        return_list, episodes_len_list = self.train_CNN_DQN(buffer_size, num_episodes, minimal_size, batch_size)
        self.plot_CNN_DQN(return_list, episodes_len_list)


class Double_DQN:
    '''
        Double DQN算法
        state_dim: int, 状态空间的维度
        action_dim: int, 动作空间的维度
        hidden_dim: int, 隐藏层的维度
        learning_rate: float, 学习率
        gamma: float, 折扣因子
        epsilon: float, ε-greedy策略中的ε
        target_update: int, 目标网络更新频率
        device: torch.device, 设备(cpu或gpu)
    '''
    def __init__(self,state_dim, action_dim, hidden_dims, learning_rate, 
                 gamma,epsilon, target_update, device,env) -> None:
        self.action_dim = action_dim
        self.main_q_net = Qnet(state_dim, action_dim, hidden_dims).to(device) # Q网络，也就是我们要训练的网络
        self.target_q_net = Qnet(state_dim, action_dim, hidden_dims).to(device) # 目标网络,就是隔一段时间更新一次
        self.optimizer = torch.optim.Adam(self.main_q_net.parameters(), lr=learning_rate) #使用Adam优化器
        self.gamma = gamma # 折扣因子
        self.epsilon = epsilon # ε-greedy策略中的ε
        self.target_update = target_update # 目标网络更新频率
        self.count = 0 # 计数器,记录更新次数
        self.device = device
        self.env = env
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            # print("动作空间是离散的")
            self.action_type = 'discrete'
        elif isinstance(self.env.action_space, gym.spaces.Box):
            # print("动作空间是连续的")
            self.action_type = 'continuous'
        else:
            # print("动作空间是其他类型")
            raise ValueError("Unsupported action space type")

    def take_action(self, state):
        '''
            ε-greedy策略选择动作
            state: np.ndarray, 当前状态
        '''
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.main_q_net(state).argmax().item()
        return action

    def best_action(self, state):
        '''
            选择最优动作
            state: np.ndarray, 当前状态
        '''
        state = torch.FloatTensor(state).to(self.device)# 将state转换为张量
        q_value = self.main_q_net(state)# 计算Q值
        action = torch.argmax(q_value).item()# 选择Q值最大的动作
        if self.action_type == 'continuous':
            action = [dis_to_con(action, self.env, self.action_dim)]
        return action
        
    def update(self,transition_dict):
        '''
            更新Q网络
            transition_dict: dict, 从经验回放池中采样的数据
        '''
        # 将数据转换为张量
        states = torch.FloatTensor(transition_dict['states']).to(self.device)
        actions = torch.LongTensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.FloatTensor(transition_dict['rewards']).view(-1, 1).to(self.device)
        next_states = torch.FloatTensor(transition_dict['next_states']).to(self.device)
        dones = torch.FloatTensor(transition_dict['dones']).view(-1, 1).to(self.device)


        # 计算Q值和目标Q值
        q_values = self.main_q_net(states).gather(1, actions)  # Q值，Q(s,a)
        # Double DQN 避免Q值过估计
        max_action = self.main_q_net(next_states).max(1)[1].view(-1, 1).long()#用主Q网络选择下一个状态的最优动作

        max_next_q_values = self.target_q_net(next_states).gather(1, max_action)#用目标Q网络计算下一个状态的最大Q值
        # DQN
        # max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)# 下个状态的最大Q值
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD误差目标, TD_target = r + γmaxQ(s',a')

        # 更新Q网络
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数,TD误差TD_error = Q_target - Q(s,a)
        self.optimizer.zero_grad() # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward() # 反向传播更新参数
        self.optimizer.step() # 更新参数

        # 每隔一段时间更新目标网络
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.main_q_net.state_dict()) # 更新目标网络
        self.count += 1


    def train_Double_DQN(self,buffer_size,num_episodes,minimal_size,batch_size):
        """ 
            运行Double DQN算法 
            buffer_size: int, 经验回放池的容量
            num_episodes: int, 训练的序列数量
            minimal_size: int, 经验回放池中数据的最小数量
            batch_size: int, 批次大小
        """
        replay_buffer = ReplayBuffer(buffer_size) # 经验回放池
        return_list = []  # 记录每一条序列的回报
        episodes_len_list = [] # 记录每一条序列的长度
        for i in range(10): #显示10个进度条
            # tqdm的进度条功能
            with tqdm(total=int(num_episodes/10) , desc='Iteration %d' % i) as pbar:
                for i_episode in range(int(num_episodes/10)):
                    st = self.env.reset()
                    done = False
                    episode_return = 0
                    episodes_len = 0
                    while not done:
                        at = self.take_action(st)
                        # print(at)
                        if self.action_type == 'continuous':
                            at_con = [dis_to_con(at, self.env, self.action_dim)]
                            st1,rt1,down_,_ = self.env.step(at_con)
                        else:
                            st1,rt1,down_,_ = self.env.step(at)
                        replay_buffer.add(st, at, rt1, st1, down_)
                        episode_return += rt1
                        st = st1
                        done = down_
                        episodes_len += 1
                        # 当buffer数据的数量超过一定值后,才进行Q网络训练
                        if replay_buffer.size() > minimal_size:
                            b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                            transition_dict = {
                                'states': b_s,
                                'actions': b_a,
                                'rewards': b_r,
                                'next_states': b_ns,
                                'dones': b_d
                            }
                            self.update(transition_dict)
                    return_list.append(episode_return)
                    episodes_len_list.append(episodes_len)
                    if (i_episode + 1) % 10 == 0:# 每10条序列打印一下这10条序列的平均回报
                        pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                        })
                    pbar.update(1)
        return return_list,episodes_len_list

    
    def plot_Double_DQN(self,return_list,episodes_len_list):
        """ 绘制Double DQN算法的学习曲线 """
        episodes_list = list(range(len(return_list)))
        plt.figure(figsize=(10, 8))
        plt.suptitle('Double DQN on {}'.format(self.env.spec.id))

        plt.subplot(3, 1, 1)
        plt.plot(episodes_list, return_list)
        plt.ylabel('Total rewards')

        mv_return = rl_utils.moving_average(return_list, 9)
        plt.subplot(3, 1, 2)
        plt.plot(episodes_list, mv_return)
        plt.ylabel('Moving average rewards')

        plt.subplot(3, 1, 3)
        plt.plot(episodes_list, episodes_len_list)
        plt.xlabel('Episodes index')
        plt.ylabel('Episodes length')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    def run_Double_DQN(self,buffer_size,num_episodes,minimal_size,batch_size):

        
        ''' 
            运行Double DQN算法
            buffer_size: int, 经验回放池的容量
            num_episodes: int, 训练的序列数量
            minimal_size: int, 经验回放池中数据的最小数量
            batch_size: int, 批次大小
        '''
        return_list,episodes_len_list = self.train_Double_DQN(buffer_size,num_episodes,minimal_size,batch_size)
        self.plot_Double_DQN(return_list,episodes_len_list)


class Dueling_DQN:
    '''
        Dueling DQN算法
        state_dim: int, 状态空间的维度
        action_dim: int, 动作空间的维度
        hidden_dim: int, 隐藏层的维度
        learning_rate: float, 学习率
        gamma: float, 折扣因子
        epsilon: float, ε-greedy策略中的ε
        target_update: int, 目标网络更新频率
        device: torch.device, 设备(cpu或gpu)
    '''
    def __init__(self,state_dim, action_dim, hidden_dim, learning_rate, 
                 gamma,epsilon, target_update, device,env) -> None:
        self.action_dim = action_dim
        self.main_q_net = VAnet(state_dim, action_dim, hidden_dim).to(device) # Q网络，也就是我们要训练的网络
        self.target_q_net = VAnet(state_dim, action_dim, hidden_dim).to(device) # 目标网络,就是隔一段时间更新一次
        self.optimizer = torch.optim.Adam(self.main_q_net.parameters(), lr=learning_rate) #使用Adam优化器
        self.gamma = gamma # 折扣因子
        self.epsilon = epsilon # ε-greedy策略中的ε
        self.target_update = target_update # 目标网络更新频率
        self.count = 0 # 计数器,记录更新次数
        self.device = device
        self.env = env
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            # print("动作空间是离散的")
            self.action_type = 'discrete'
        elif isinstance(self.env.action_space, gym.spaces.Box):
            # print("动作空间是连续的")
            self.action_type = 'continuous'
        else:
            # print("动作空间是其他类型")
            raise ValueError("Unsupported action space type")

    def take_action(self, state):
        '''
            ε-greedy策略选择动作
            state: np.ndarray, 当前状态
        '''
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.main_q_net(state).argmax().item()
        return action

    def best_action(self, state):
        '''
            选择最优动作
            state: np.ndarray, 当前状态
        '''
        state = torch.FloatTensor(state).to(self.device)# 将state转换为张量
        q_value = self.main_q_net(state)# 计算Q值
        action = torch.argmax(q_value).item()# 选择Q值最大的动作
        if self.action_type == 'continuous':
            action = [dis_to_con(action, self.env, self.action_dim)]
        return action
        
    def update(self,transition_dict):
        '''
            更新Q网络
            transition_dict: dict, 从经验回放池中采样的数据
        '''
        # 将数据转换为张量
        states = torch.FloatTensor(transition_dict['states']).to(self.device)
        actions = torch.LongTensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.FloatTensor(transition_dict['rewards']).view(-1, 1).to(self.device)
        next_states = torch.FloatTensor(transition_dict['next_states']).to(self.device)
        dones = torch.FloatTensor(transition_dict['dones']).view(-1, 1).to(self.device)

        # 计算Q值和目标Q值
        q_values = self.main_q_net(states).gather(1, actions)  # Q值，Q(s,a)
        # Dueling DQN
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)# 下个状态的最大Q值
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD误差目标, TD_target = r + γmaxQ(s',a')

        # 更新Q网络
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数,TD误差TD_error = Q_target - Q(s,a)
        self.optimizer.zero_grad() # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward() # 反向传播更新参数
        self.optimizer.step() # 更新参数

        # 每隔一段时间更新目标网络
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.main_q_net.state_dict()) # 更新目标网络
        self.count += 1


    def train_Dueling_DQN(self,buffer_size,num_episodes,minimal_size,batch_size):
        """ 
            运行Dueling DQN算法 
            buffer_size: int, 经验回放池的容量
            num_episodes: int, 训练的序列数量
            minimal_size: int, 经验回放池中数据的最小数量
            batch_size: int, 批次大小
        """
        replay_buffer = ReplayBuffer(buffer_size) # 经验回放池
        return_list = []  # 记录每一条序列的回报
        episodes_len_list = [] # 记录每一条序列的长度
        for i in range(10): #显示10个进度条
            # tqdm的进度条功能
            with tqdm(total=int(num_episodes/10) , desc='Iteration %d' % i) as pbar:
                for i_episode in range(int(num_episodes/10)):
                    st = self.env.reset()
                    done = False
                    episode_return = 0
                    episodes_len = 0
                    while not done:
                        at = self.take_action(st)
                        # print(at)
                        if self.action_type == 'continuous':
                            at_con = [dis_to_con(at, self.env, self.action_dim)]
                            st1,rt1,down_,_ = self.env.step(at_con)
                        else:
                            st1,rt1,down_,_ = self.env.step(at)
                        replay_buffer.add(st, at, rt1, st1, down_)
                        episode_return += rt1
                        st = st1
                        done = down_
                        episodes_len += 1
                        # 当buffer数据的数量超过一定值后,才进行Q网络训练
                        if replay_buffer.size() > minimal_size:
                            b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                            transition_dict = {
                                'states': b_s,
                                'actions': b_a,
                                'rewards': b_r,
                                'next_states': b_ns,
                                'dones': b_d
                            }
                            self.update(transition_dict)
                    return_list.append(episode_return)
                    episodes_len_list.append(episodes_len)
                    if (i_episode + 1) % 10 == 0:# 每10条序列打印一下这10条序列的平均回报
                        pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                        })
                    pbar.update(1)
        return return_list,episodes_len_list

    
    def plot_Dueling_DQN(self,return_list,episodes_len_list):
        """ 绘制 Dueling DQN算法的学习曲线 """
        episodes_list = list(range(len(return_list)))
        plt.figure(figsize=(10, 8))
        plt.suptitle('Dueling DQN on {}'.format(self.env.spec.id))

        plt.subplot(3, 1, 1)
        plt.plot(episodes_list, return_list)
        plt.ylabel('Total rewards')

        mv_return = rl_utils.moving_average(return_list, 9)
        plt.subplot(3, 1, 2)
        plt.plot(episodes_list, mv_return)
        plt.ylabel('Moving average rewards')

        plt.subplot(3, 1, 3)
        plt.plot(episodes_list, episodes_len_list)
        plt.xlabel('Episodes index')
        plt.ylabel('Episodes length')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    def run_Dueling_DQN(self,buffer_size,num_episodes,minimal_size,batch_size):
        
        ''' 
            运行Dueling DQN算法
            buffer_size: int, 经验回放池的容量
            num_episodes: int, 训练的序列数量
            minimal_size: int, 经验回放池中数据的最小数量
            batch_size: int, 批次大小
        '''
        return_list,episodes_len_list = self.train_Dueling_DQN(buffer_size,num_episodes,minimal_size,batch_size)
        self.plot_Dueling_DQN(return_list,episodes_len_list)
