"""
    Policy gradient 算法
    

    代码库中包括 REINFORCE

    Author: Z.R.Wang
    Aug,26,2024 in YSU
    Emile: wangzhanran@stumail.ysu.edu.cn

    reference:强化学习的数学原理(西湖大学赵世钰老师),动手学强化学习https://github.com/boyu-ai/Hands-on-RL
"""


import numpy as np
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
    return re


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims):
        super(PolicyNet, self).__init__()
        self.hidden_layers = torch.nn.ModuleList()
        
        # 输入层到第一个隐藏层
        self.hidden_layers.append(torch.nn.Linear(state_dim, hidden_dims[0]))
        
        # 隐藏层之间
        for i in range(1, len(hidden_dims)):
            self.hidden_layers.append(torch.nn.Linear(hidden_dims[i-1], hidden_dims[i]))
        
        # 最后一个隐藏层到输出层
        self.output_layer = torch.nn.Linear(hidden_dims[-1], action_dim)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        return F.softmax(self.output_layer(x), dim=1)
    
class REINFORCE:
    def __init__(self, state_dim, action_dim, hidden_dims, learning_rate, 
                 gamma , device, num_episodes, env) -> None:
        self.policy_net = PolicyNet(state_dim, action_dim, hidden_dims).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate) # 使用Adam优化器
        self.gamma = gamma # 折扣因子
        self.device = device
        self.num_episodes = num_episodes
        self.env = env
        self.action_dim = action_dim
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
        # 根据动作概率分布随机采样
        state = torch.tensor([state], dtype=torch.float).to(self.device)  # 将状态转换为张量并发送到设备
        probs = self.policy_net(state)  # 获取动作概率分布
        action_dist = torch.distributions.Categorical(probs)  # 创建分类分布对象
        action = action_dist.sample()  # 从分布中采样动作
        return action.item()  # 返回采样的动作

    def update(self, transition_dict):
        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']

        # 计算折扣回报
        G = 0;
        self.optimizer.zero_grad()
        for i in reversed(range(len(reward_list))): # 从最后一步算起
            G = self.gamma * G + reward_list[i]
            state = torch.tensor([state_list[i]],
                                 dtype=torch.float).to(self.device)
            action = torch.tensor([action_list[i]]).view(-1,1).to(self.device)
            log_prob = torch.log(self.policy_net(state).gather(1, action))
            loss = -log_prob * G  # 每一步的损失函数
            loss.backward() # 反向传播计算梯度
        self.optimizer.step() # 梯度下降

    def train_REINFORCE(self):
        """ 
            运行 REINFORCE 算法 
        """
 
        return_list = []  # 记录每一条序列的回报
        episodes_len_list = [] # 记录每一条序列的长度
        for i in range(10): #显示10个进度条
            # tqdm的进度条功能
            with tqdm(total=int(self.num_episodes/10) , desc='Iteration %d' % i) as pbar:
                for i_episode in range(int(self.num_episodes/10)):
                    st = self.env.reset()
                    done = False
                    episode_return = 0
                    episodes_len = 0
                    transition_dict = {
                        'states': [],   # 状态
                        'actions': [],  # 动作
                        'next_states': [],  # 下一个状态
                        'rewards': [],  # 回报
                        'dones': [] # 是否结束
                    }
                    while not done:
                        at = self.take_action(st)
                        if self.action_type == 'continuous':
                            at_con = [dis_to_con(at, self.env, self.action_dim)]
                            st1,rt1,down_,_ = self.env.step(at_con)
                        else:
                            st1,rt1,down_,_ = self.env.step(at)
                        transition_dict['states'].append(st)
                        transition_dict['actions'].append(at)
                        transition_dict['next_states'].append(st1)
                        transition_dict['rewards'].append(rt1)
                        transition_dict['dones'].append(down_)
                        episode_return += rt1
                        st = st1
                        done = down_
                        episodes_len += 1
                    return_list.append(episode_return)
                    episodes_len_list.append(episodes_len)
                    self.update(transition_dict)
                    if (i_episode + 1) % 10 == 0:# 每10条序列打印一下这10条序列的平均回报
                        pbar.set_postfix({
                        'episode':
                        '%d' % (self.num_episodes / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                        })
                    pbar.update(1)
        return return_list,episodes_len_list

    
    def plot_REINFORCE(self,return_list,episodes_len_list):
        """ 绘制 REINFORCE 算法的学习曲线 """
        episodes_list = list(range(len(return_list)))
        plt.figure(figsize=(10, 8))
        plt.suptitle('REINFORCE on {}'.format(self.env.spec.id))

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

    def run_REINFORCE(self):
        
        ''' 
            运行REINFORCE算法
        '''
        return_list,episodes_len_list = self.train_REINFORCE()
        self.plot_REINFORCE(return_list,episodes_len_list)