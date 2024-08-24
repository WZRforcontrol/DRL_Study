import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random
import time

class DynaQ:
    """ Dyna-Q算法 """
    def __init__(self, epsilon, alpha, gamma, s_num, n_action, n_planning, env, num_episodes):
        self.Q_table = np.zeros([s_num, n_action])  # 初始化Q(s,a)表格 
        self.n_action = n_action  # 动作个数
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略中的参数
        self.env = env
        self.num_episodes = num_episodes # 智能体在环境中运行的序列的数量
        self.n_planning = n_planning  #执行Q-planning的次数, 对应1次Q-learning
        self.model = dict()  # 环境模型

    def take_action(self, state):  # 选取下一步的操作
        if np.random.random() < self.epsilon:
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
    
    def q_learning(self, st, at, rt1, st1):
        td_error = rt1 + self.gamma * self.Q_table[st1].max(
        ) - self.Q_table[st, at]
        self.Q_table[st, at] += self.alpha * td_error

    def update(self, st, at, rt1, st1):
        self.q_learning(st, at, rt1, st1)
        self.model[(st, at)] = rt1, st1  # 将数据添加到模型中
        for _ in range(self.n_planning):  # Q-planning循环
            # 随机选择曾经遇到过的状态动作对
            (s, a), (rt1, s_) = random.choice(list(self.model.items()))
            self.q_learning(s, a, rt1, s_)

    def train_DynaQ(self):
        """ 运行DynaQ算法 """
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
    
    def plot_DynaQ(self,return_list,episodes_len_list):
        """ 绘制DynaQ算法的学习曲线 """
        episodes_list = list(range(len(return_list)))
        plt.figure(figsize=(10, 8))
        plt.suptitle('DynaQ on {}'.format(self.env.spec.id))

        plt.subplot(2, 1, 1)
        plt.plot(episodes_list, return_list)
        plt.ylabel('Total rewards')

        plt.subplot(2, 1, 2)
        plt.plot(episodes_list, episodes_len_list)
        plt.xlabel('Episodes index')
        plt.ylabel('Episodes length')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    def run_DynaQ(self):
        return_list,episodes_len_list = self.train_DynaQ()
        self.plot_DynaQ(return_list,episodes_len_list)