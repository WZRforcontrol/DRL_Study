"""
    TD learning based on table 算法
    时序差分学习算法是一种无模型的强化学习算法，它通过估计值函数来学习策略。
    TD学习算法是一种在线学习算法，它在每一步都更新值函数的估计，而不是等到一条序列结束后再更新。
    TD学习算法的一个重要特点是它可以在没有环境模型的情况下学习策略，这使得它可以应用于很多实际问题中。
    TD学习算法的一个重要应用是Q-learning算法，它是一种基于值函数的强化学习算法，可以用来学习最优策略。
    TD学习算法的另一个重要应用是Sarsa算法，它是一种基于值函数的强化学习算法，可以用来学习最优策略。

    代码库中包括 Sarsa , Expected Sarsa , n step Sarsa , Q learning

    Author: Z.R.Wang
    Aug,22-23,2024 in YSU
    Emile: wangzhanran@stumail.ysu.edu.cn

    reference:强化学习的数学原理(西湖大学赵世钰老师),动手学强化学习https://github.com/boyu-ai/Hands-on-RL
"""

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

class Sarsa:
    """ Sarsa算法 """
    def __init__(self, epsilon, alpha, gamma, s_num, n_action, env, num_episodes):
        self.Q_table = np.zeros([s_num, n_action])  # 初始化Q(s,a)表格 
        self.n_action = n_action  # 动作个数
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略中的参数
        self.env = env
        self.num_episodes = num_episodes # 智能体在环境中运行的序列的数量


    def take_action(self,state):
        '''选取下一步的操作,具体实现为epsilon-贪婪策略'''
        if np.random.random() < self.epsilon:  # 轮盘赌
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])

        return action
    
    def best_action(self,state):
        '''用于打印策略'''
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):
            if self.Q_table[state][i] == Q_max:
                a[i] = 1        
        return a#返回最优动作的列表,最优动作对应的位置为1，其余为0
    
    def update(self, st, at, rt1, st1, at1):
        """ 更新Q(s,a) """
        # Q(s,a) = Q(s,a) - alpha * ( Q(s,a) - (r + gamma * Q(s',a')) )
        self.Q_table[st][at] -= self.alpha * (self.Q_table[st][at] - (rt1 + self.gamma * self.Q_table[st1][at1] ))

    def train_Sarsa(self):
        """ 运行Sarsa算法 """
        return_list = []  # 记录每一条序列的回报
        episodes_len_list = [] # 记录每一条序列的长度
        for i in range(10): #显示10个进度条
            # tqdm的进度条功能
            with tqdm(total=int(self.num_episodes/10) , desc='Iteration %d' % i) as pbar:
                for i_episode in range(int(self.num_episodes/10)):
                    episode_return = 0
                    st = self.env.reset()
                    at = self.take_action(st)
                    done = False
                    episodes_len = 0
                    while not done:
                        st1,rt1,down_,info = self.env.step(at)
                        at1 = self.take_action(st1)
                        episode_return += rt1
                        self.update(st, at, rt1, st1, at1)
                        st = st1
                        at = at1
                        done = down_
                        episodes_len += 1
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
    
    def plot_Sarsa(self,return_list,episodes_len_list):
        """ 绘制Sarsa算法的学习曲线 """
        episodes_list = list(range(len(return_list)))
        plt.figure(figsize=(10, 8))
        plt.suptitle('Sarsa on {}'.format(self.env.spec.id))

        plt.subplot(2, 1, 1)
        plt.plot(episodes_list, return_list)
        plt.ylabel('Total rewards')

        plt.subplot(2, 1, 2)
        plt.plot(episodes_list, episodes_len_list)
        plt.xlabel('Episodes index')
        plt.ylabel('Episodes length')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    def run_Sarsa(self):
        return_list,episodes_len_list = self.train_Sarsa()
        self.plot_Sarsa(return_list,episodes_len_list)
        

class Expected_Sarsa:
    """ Expected Sarsa算法 """
    def __init__(self, epsilon, alpha, gamma, s_num, n_action, env, num_episodes):
        self.Q_table = np.zeros([s_num, n_action])  # 初始化Q(s,a)表格 
        self.n_action = n_action  # 动作个数
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略中的参数
        self.env = env
        self.num_episodes = num_episodes # 智能体在环境中运行的序列的数量


    def take_action(self,state):
        '''选取下一步的操作,具体实现为epsilon-贪婪策略'''
        if np.random.random() < self.epsilon:  # 轮盘赌
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])

        return action
    
    def best_action(self,state):
        '''用于打印策略'''
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):
            if self.Q_table[state][i] == Q_max:
                a[i] = 1        
        return a#返回最优动作的列表,最优动作对应的位置为1，其余为0
    
    def update(self, st, at, rt1, st1, at1):
        """ 更新Q(s,a) """
        # 计算 self.Q_table[st1] 的期望值
        expected_q_st1 = np.mean(self.Q_table[st1])
        # Q(s,a) = Q(s,a) - alpha * ( Q(s,a) - (r + gamma * E [Q(s',A)]) )
        self.Q_table[st][at] -= self.alpha * (self.Q_table[st][at] - (rt1 + self.gamma * expected_q_st1))

    def train_Expected_Sarsa(self):
        """ 运行Expected Sarsa算法 """
        return_list = []  # 记录每一条序列的回报
        episodes_len_list = [] # 记录每一条序列的长度
        for i in range(10): #显示10个进度条
            # tqdm的进度条功能
            with tqdm(total=int(self.num_episodes/10) , desc='Iteration %d' % i) as pbar:
                for i_episode in range(int(self.num_episodes/10)):
                    episode_return = 0
                    st = self.env.reset()
                    at = self.take_action(st)
                    done = False
                    episodes_len = 0
                    while not done:
                        st1,rt1,down_,info = self.env.step(at)
                        at1 = self.take_action(st1)
                        episode_return += rt1
                        self.update(st, at, rt1, st1, at1)
                        st = st1
                        at = at1
                        done = down_
                        episodes_len += 1
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
    
    def plot_Expected_Sarsa(self,return_list,episodes_len_list):
        """ 绘制Expected Sarsa算法的学习曲线 """
        episodes_list = list(range(len(return_list)))
        plt.figure(figsize=(10, 8))
        plt.suptitle('Expected Sarsa on {}'.format(self.env.spec.id))

        plt.subplot(2, 1, 1)
        plt.plot(episodes_list, return_list)
        plt.ylabel('Total rewards')

        plt.subplot(2, 1, 2)
        plt.plot(episodes_list, episodes_len_list)
        plt.xlabel('Episodes index')
        plt.ylabel('Episodes length')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    def run_Expected_Sarsa(self):
        return_list,episodes_len_list = self.train_Expected_Sarsa()
        self.plot_Expected_Sarsa(return_list,episodes_len_list)
        

class nstep_Sarsa:
    """ n step Sarsa 算法 """
    def __init__(self, epsilon, alpha, gamma, n, s_num, n_action, env, num_episodes):
        self.Q_table = np.zeros([s_num, n_action])  # 初始化Q(s,a)表格 
        self.n_action = n_action  # 动作个数
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略中的参数
        self.env = env
        self.num_episodes = num_episodes # 智能体在环境中运行的序列的数量
        self.n = n # 采用 n 步Sarsa算法
        self.state_list = []  # 保存之前的状态
        self.action_list = []  # 保存之前的动作
        self.reward_list = []  # 保存之前的奖励


    def take_action(self,state):
        '''选取下一步的操作,具体实现为epsilon-贪婪策略'''
        if np.random.random() < self.epsilon:  # 轮盘赌
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])

        return action
    
    def best_action(self,state):
        '''用于打印策略'''
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):
            if self.Q_table[state][i] == Q_max:
                a[i] = 1        
        return a#返回最优动作的列表,最优动作对应的位置为1，其余为0
    
    def update(self, st, at, rt1, st1, at1, done):
        self.state_list.append(st)
        self.action_list.append(at)
        self.reward_list.append(rt1)
        if len(self.state_list) == self.n:  # 若保存的数据可以进行n步更新
            G = self.Q_table[st1, at1]  # 得到Q(s_{t+n}, a_{t+n})
            for i in reversed(range(self.n)):
                G = self.reward_list[i] + self.gamma * G # 不断向前计算每一步的回报
                # 如果到达终止状态,最后几步虽然长度不够n步,也将其进行更新
                if done and i > 0:
                    s = self.state_list[i]
                    a = self.action_list[i]
                    self.Q_table[s,a] -= self.alpha * (self.Q_table[s,a] - G)
            s = self.state_list.pop(0)  # 将需要更新的状态动作从列表中删除,下次不必更新
            a = self.action_list.pop(0)
            self.reward_list.pop(0)
            """ 更新Q(s,a) """
            # G = r_{t} + gamma * r_{t+1} + ... + gamma^{n} * Q(s_{t+n},a_{t+n})
            # Q(s,a) = Q(s,a) - alpha * ( Q(s,a) - G )
            self.Q_table[s,a] -= self.alpha * (self.Q_table[s,a] - G)
        if done:  # 如果到达终止状态,即将开始下一条序列,则将列表全清空
            self.state_list = []
            self.action_list = []
            self.reward_list = []

    def train_nstep_Sarsa(self):
        """ 运行 n step Sarsa算法 """
        return_list = []  # 记录每一条序列的回报
        episodes_len_list = [] # 记录每一条序列的长度
        for i in range(10): #显示10个进度条
            # tqdm的进度条功能
            with tqdm(total=int(self.num_episodes/10) , desc='Iteration %d' % i) as pbar:
                for i_episode in range(int(self.num_episodes/10)):
                    episode_return = 0
                    st = self.env.reset()
                    at = self.take_action(st)
                    done = False
                    episodes_len = 0
                    while not done:
                        st1,rt1,down_,info = self.env.step(at)
                        at1 = self.take_action(st1)
                        episode_return += rt1
                        self.update(st, at, rt1, st1, at1, down_)
                        st = st1
                        at = at1
                        done = down_
                        episodes_len += 1
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
    
    def plot_nstep_Sarsa(self,return_list,episodes_len_list):
        """ 绘制 n step Sarsa算法的学习曲线 """
        episodes_list = list(range(len(return_list)))
        plt.figure(figsize=(10, 8))
        plt.suptitle('n step Sarsa on {}'.format(self.env.spec.id))

        plt.subplot(2, 1, 1)
        plt.plot(episodes_list, return_list)
        plt.ylabel('Total rewards')

        plt.subplot(2, 1, 2)
        plt.plot(episodes_list, episodes_len_list)
        plt.xlabel('Episodes index')
        plt.ylabel('Episodes length')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    def run_nstep_Sarsa(self):
        return_list,episodes_len_list = self.train_nstep_Sarsa()
        self.plot_nstep_Sarsa(return_list,episodes_len_list)


class Q_learning:
    """ Q learning算法 """
    def __init__(self, epsilon, alpha, gamma, s_num, n_action, env, num_episodes):
        self.Q_table = np.zeros([s_num, n_action])  # 初始化Q(s,a)表格 
        self.n_action = n_action  # 动作个数
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略中的参数
        self.env = env
        self.num_episodes = num_episodes # 智能体在环境中运行的序列的数量
    
    def take_action(self,state): #选取下一步的操作
        # 由于 Q learning 是一种离线策略算法。
        # 其中一条策略为行为策略。用来采样数据,负责探索。
        # 而另一条策略为目标策略。用来更新Q值,负责利用。
        # 因此第二条策略不需要具有探索性。因此无需使用epsilon-贪婪策略贪心策略。
        # 赵世钰老师所讲,存疑
        if np.random.random() < self.epsilon:  # 轮盘赌
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action
    
    def best_action(self, state):  # 用于打印策略
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):
            if self.Q_table[state, i] == Q_max:
                a[i] = 1
        return a
    
    def best_action(self,state):
        '''用于打印策略'''
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):
            if self.Q_table[state][i] == Q_max:
                a[i] = 1        
        return a#返回最优动作的列表,最优动作对应的位置为1，其余为0
    
    def update(self, st, at, rt1, st1):
        """ 更新Q(s,a) """
        # Q(s,a) = Q(s,a) - alpha * ( Q(s,a) - (r + gamma * argmax_a'^(Q(s',A))) )
        self.Q_table[st][at] -= self.alpha * (self.Q_table[st][at] - (rt1 + self.gamma * np.max(self.Q_table[st1])))

    def train_Q_learning(self):
        """ 运行Q learning算法 """
        return_list = []  # 记录每一条序列的回报
        episodes_len_list = [] # 记录每一条序列的长度
        for i in range(10): #显示10个进度条
            # tqdm的进度条功能
            with tqdm(total=int(self.num_episodes/10) , desc='Iteration %d' % i) as pbar:
                for i_episode in range(int(self.num_episodes/10)):
                    episode_return = 0
                    st = self.env.reset()
                    done = False
                    episodes_len = 0
                    while not done:
                        at = self.take_action(st)
                        st1,rt1,down_,info = self.env.step(at)
                        episode_return += rt1
                        self.update(st, at, rt1, st1)
                        st = st1
                        done = down_
                        episodes_len += 1
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
    
    def plot_Q_learning(self,return_list,episodes_len_list):
        """ 绘制 Q learning 算法的学习曲线 """
        episodes_list = list(range(len(return_list)))
        plt.figure(figsize=(10, 8))
        plt.suptitle('Q learning on {}'.format(self.env.spec.id))

        plt.subplot(2, 1, 1)
        plt.plot(episodes_list, return_list)
        plt.ylabel('Total rewards')

        plt.subplot(2, 1, 2)
        plt.plot(episodes_list, episodes_len_list)
        plt.xlabel('Episodes index')
        plt.ylabel('Episodes length')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    def run_Q_learning(self):
        return_list,episodes_len_list = self.train_Q_learning()
        self.plot_Q_learning(return_list,episodes_len_list)
          