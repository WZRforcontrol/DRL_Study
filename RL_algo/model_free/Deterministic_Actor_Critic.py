"""
    Actor Critic method
    

    代码库中包括 DDPG SAC 算法的实现

    Author: Z.R.Wang
    Aug,29,2024 in YSU
    Emile: wangzhanran@stumail.ysu.edu.cn

    reference:强化学习的数学原理(西湖大学赵世钰老师),动手学强化学习https://github.com/boyu-ai/Hands-on-RL
"""

import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
import RL_algo.rl_utils as rl_utils
import gym
import copy
import collections
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done): 
        self.buffer.append((state, action, reward, next_state, done)) 

    def sample(self, batch_size): 
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done 

    def size(self): 
        return len(self.buffer)
    

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims, action_bound):
        super(PolicyNet, self).__init__()
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(torch.nn.Linear(input_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            input_dim = hidden_dim
        layers.append(torch.nn.Linear(input_dim, action_dim))
        layers.append(torch.nn.Tanh())
        self.model = torch.nn.Sequential(*layers)
        self.action_bound = action_bound

    def forward(self, x):
        return self.model(x) * self.action_bound


class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims):
        super(QValueNet, self).__init__()
        layers = []
        input_dim = state_dim + action_dim
        for hidden_dim in hidden_dims:
            layers.append(torch.nn.Linear(input_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            input_dim = hidden_dim
        layers.append(torch.nn.Linear(input_dim, 1))
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1) # 拼接状态和动作
        return self.model(cat)

class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims, action_bound):
        super(PolicyNetContinuous, self).__init__()
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(torch.nn.Linear(input_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            input_dim = hidden_dim
        self.hidden_layers = torch.nn.Sequential(*layers)
        self.fc_mu = torch.nn.Linear(input_dim, action_dim)
        self.fc_std = torch.nn.Linear(input_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, x):
        x = self.hidden_layers(x)
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        dist = Normal(mu, std)
        normal_sample = dist.rsample()  # rsample()是重参数化采样
        log_prob = dist.log_prob(normal_sample)
        action = torch.tanh(normal_sample)
        # 计算tanh_normal分布的对数概率密度
        log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        action = action * self.action_bound
        return action, log_prob


class QValueNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims):
        super(QValueNetContinuous, self).__init__()
        layers = []
        input_dim = state_dim + action_dim
        for hidden_dim in hidden_dims:
            layers.append(torch.nn.Linear(input_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            input_dim = hidden_dim
        layers.append(torch.nn.Linear(input_dim, 1))
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        return self.model(cat)
    

class DDPG:
    '''
        DDPG算法
    '''
    def __init__(self, state_dim, action_dim, hidden_dims, action_bound, 
                 sigma, actor_lr, critic_lr, tau, gamma, device, num_episodes, env) -> None:
        # 网络定义
        # 策略网络
        self.actor = PolicyNet(state_dim, action_dim, hidden_dims, action_bound).to(device) 
        # 价值网络
        self.critic = QValueNet(state_dim, action_dim, hidden_dims).to(device)
        # 目标策略网络
        self.target_actor = PolicyNet(state_dim, action_dim, hidden_dims, action_bound).to(device) 
        # 目标价值网络
        self.target_critic = QValueNet(state_dim, action_dim, hidden_dims).to(device)
        # 目标策略网络和策略网络参数相同
        self.target_actor.load_state_dict(self.actor.state_dict())
        # 目标价值网络和价值网络参数相同
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr) # 策略网络优化器
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr) # 价值网络优化器

        # 参数
        self.sigma = sigma  # 高斯噪声的标准差,均值直接设为0
        self.tau = tau # 目标网络软更新参数
        self.gamma = gamma # 折扣因子
        self.device = device
        self.num_episodes = num_episodes    
        self.env = env 
        self.action_dim = action_dim

    def take_action(self, state):
        state = torch.FloatTensor([state]).to(self.device)  # 再转换为张量
        action = self.actor(state).item()
        # 给动作添加噪声，增加探索
        action = action + self.sigma * np.random.randn(self.action_dim)
        return action

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        # 将数据转换为张量
        states = torch.FloatTensor(transition_dict['states']).to(self.device)
        actions = torch.FloatTensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.FloatTensor(transition_dict['rewards']).view(-1, 1).to(self.device)
        next_states = torch.FloatTensor(transition_dict['next_states']).to(self.device)
        dones = torch.FloatTensor(transition_dict['dones']).view(-1, 1).to(self.device)
        
        rewards = (rewards + 8.0) / 8.0  # 和TRPO一样,对奖励进行修改,方便训练

        #  
        next_q_values = self.target_critic(next_states, self.target_actor(next_states))
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)
        critic_loss = torch.mean(F.mse_loss(self.critic(states, actions), q_targets))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -torch.mean(self.critic(states, self.actor(states)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.actor, self.target_actor)  # 软更新策略网络
        self.soft_update(self.critic, self.target_critic)  # 软更新价值网络

    def train_DDPG(self, buffer_size, minimal_size, batch_size):
        """ 
            运行 DDPG 算法 
            buffer_size: int, 经验回放池的容量
            minimal_size: int, 经验回放池中数据的最小数量
            batch_size: int, 批次大小
        """
        replay_buffer = rl_utils.ReplayBuffer(buffer_size)
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
                    while not done:
                        at = self.take_action(st)
                        st1,rt1,down_,_ = self.env.step(at)
                        replay_buffer.add(st, at, rt1, st1, down_)
                        st = st1
                        done = down_
                        episode_return += rt1
                        episodes_len += 1
                        if replay_buffer.size() > minimal_size:
                            b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                            transition_dict = {'states': b_s, 
                                               'actions': b_a, 
                                               'next_states': b_ns, 
                                               'rewards': b_r, 
                                               'dones': b_d}
                            self.update(transition_dict)
                    return_list.append(episode_return)
                    episodes_len_list.append(episodes_len)
                    if (i_episode + 1) % 10 == 0:# 每10条序列打印一下这10条序列的平均回报
                        pbar.set_postfix({
                        'episode':
                        '%d' % (self.num_episodes / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                        })
                    pbar.update(1)
        return return_list,episodes_len_list

    
    def plot_DDPG(self,return_list,episodes_len_list):
        """ 绘制 DDPG 算法的学习曲线 """
        episodes_list = list(range(len(return_list)))
        plt.figure(figsize=(10, 8))
        plt.suptitle('DDPG on {}'.format(self.env.spec.id))

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

    def run_DDPG(self, buffer_size, minimal_size, batch_size):
        
        ''' 
            运行 DDPG 算法
        '''
        return_list,episodes_len_list = self.train_DDPG(buffer_size, minimal_size, batch_size)
        self.plot_DDPG(return_list,episodes_len_list)


class SACContinuous:
    ''' 处理连续动作的SAC算法 '''
    def __init__(self, state_dim, action_dim, hidden_dims, action_bound,
                 actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma,
                 device, num_episodes, env):
        self.actor = PolicyNetContinuous(state_dim, action_dim, hidden_dims,
                                         action_bound).to(device)  # 策略网络
        self.critic_1 = QValueNetContinuous(state_dim, action_dim, hidden_dims).to(device)  # 第一个Q网络
        self.critic_2 = QValueNetContinuous(state_dim, action_dim, hidden_dims).to(device)  # 第二个Q网络
        self.target_critic_1 = QValueNetContinuous(state_dim, action_dim, hidden_dims).to(device)  # 第一个目标Q网络
        self.target_critic_2 = QValueNetContinuous(state_dim, action_dim, hidden_dims).to(device)  # 第二个目标Q网络
        # 令目标Q网络的初始参数和Q网络一样
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(),
                                                   lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(),
                                                   lr=critic_lr)
        # 使用alpha的log值,可以使训练结果比较稳定
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True  # 可以对alpha求梯度
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr)
        self.target_entropy = target_entropy  # 目标熵的大小
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.num_episodes = num_episodes
        self.env = env

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action = self.actor(state)[0]
        return [action.item()]

    def calc_target(self, rewards, next_states, dones):  # 计算目标Q值
        next_actions, log_prob = self.actor(next_states)
        entropy = -log_prob
        q1_value = self.target_critic_1(next_states, next_actions)
        q2_value = self.target_critic_2(next_states, next_actions)
        next_value = torch.min(q1_value,
                               q2_value) + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        # 和之前章节一样,对倒立摆环境的奖励进行重塑以便训练
        rewards = (rewards + 8.0) / 8.0

        # 更新两个Q网络
        td_target = self.calc_target(rewards, next_states, dones)
        critic_1_loss = torch.mean(
            F.mse_loss(self.critic_1(states, actions), td_target.detach()))
        critic_2_loss = torch.mean(
            F.mse_loss(self.critic_2(states, actions), td_target.detach()))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # 更新策略网络
        new_actions, log_prob = self.actor(states)
        entropy = -log_prob
        q1_value = self.critic_1(states, new_actions)
        q2_value = self.critic_2(states, new_actions)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy -
                                torch.min(q1_value, q2_value))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新alpha值
        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)

    def train_SACCon(self, buffer_size, minimal_size, batch_size):
        """ 
            运行 SACContinuous 算法 
            buffer_size: int, 经验回放池的容量
            minimal_size: int, 经验回放池中数据的最小数量
            batch_size: int, 批次大小
        """
        replay_buffer = rl_utils.ReplayBuffer(buffer_size)
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
                    while not done:
                        at = self.take_action(st)
                        st1,rt1,down_,_ = self.env.step(at)
                        replay_buffer.add(st, at, rt1, st1, down_)
                        st = st1
                        done = down_
                        episode_return += rt1
                        episodes_len += 1
                        if replay_buffer.size() > minimal_size:
                            b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                            transition_dict = {'states': b_s, 
                                               'actions': b_a, 
                                               'next_states': b_ns, 
                                               'rewards': b_r, 
                                               'dones': b_d}
                            self.update(transition_dict)
                    return_list.append(episode_return)
                    episodes_len_list.append(episodes_len)
                    if (i_episode + 1) % 10 == 0:# 每10条序列打印一下这10条序列的平均回报
                        pbar.set_postfix({
                        'episode':
                        '%d' % (self.num_episodes / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                        })
                    pbar.update(1)
        return return_list,episodes_len_list

    
    def plot_SACCon(self,return_list,episodes_len_list):
        """ 绘制 SACContinuous 算法的学习曲线 """
        episodes_list = list(range(len(return_list)))
        plt.figure(figsize=(10, 8))
        plt.suptitle('SAC Continuous on {}'.format(self.env.spec.id))

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

    def run_SACCon(self, buffer_size, minimal_size, batch_size):
        
        ''' 
            运行 SACContinuous 算法
        '''
        return_list,episodes_len_list = self.train_SACCon(buffer_size, minimal_size, batch_size)
        self.plot_SACCon(return_list,episodes_len_list)


 