import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'E:\Anaconda\Projects\RL\Study'))
import random
import gym
import numpy as np
import torch
import highway_env
import time
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import RL_algo.rl_utils as rl_utils
import datetime
import warnings
warnings.filterwarnings("ignore")

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
    
class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dims):
        super(ValueNet, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(state_dim, hidden_dims[0]))
        for i in range(1, len(hidden_dims)):
            self.layers.append(torch.nn.Linear(hidden_dims[i-1], hidden_dims[i]))
        self.layers.append(torch.nn.Linear(hidden_dims[-1], 1))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)
    
class ActorCritic:
    def __init__(self, state_dim, action_dim, hidden_dims, actor_lr, critic_lr, 
                 gamma, device, num_episodes, env) -> None:
        # 策略网络
        self.actor = PolicyNet(state_dim, action_dim, hidden_dims).to(device) 
        # 价值网络
        self.critic = ValueNet(state_dim, hidden_dims).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)# 策略网络优化器
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr) # 价值网络优化器
        self.gamma = gamma
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
        state = torch.FloatTensor([state]).to(self.device)  # 再转换为张量
        probs = self.actor(state)  # 得到动作概率
        action_dist = torch.distributions.Categorical(probs)  # 生成一个分布, 以便采样
        action = action_dist.sample()  # 从分布中采样一个动作
        return action.item()  # 返回采样的动作
    
    def update(self, transition_dict):
        # 将数据转换为张量
        states = torch.FloatTensor(transition_dict['states']).to(self.device)
        actions = torch.LongTensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.FloatTensor(transition_dict['rewards']).view(-1, 1).to(self.device)
        next_states = torch.FloatTensor(transition_dict['next_states']).to(self.device)
        dones = torch.FloatTensor(transition_dict['dones']).view(-1, 1).to(self.device)
        
        #  
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones) # 计算TD目标
        td_error = td_target - self.critic(states) # 计算TD误差
        
        log_probs = torch.log(self.actor(states).gather(1,actions)) # 计算动作的对数概率
        actor_loss = -torch.mean(log_probs * td_error.detach()) # 计算策略损失

        critic_loss = F.mse_loss(self.critic(states), td_target.detach()) # 计算价值网络损失

        # 更新策略网络和价值网络
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def train_AC(self):
        """ 
            运行 Actor Critic 算法 
        """
 
        return_list = []  # 记录每一条序列的回报
        episodes_len_list = [] # 记录每一条序列的长度
        for i in range(10): #显示10个进度条
            # tqdm的进度条功能
            with tqdm(total=int(self.num_episodes/10) , desc='Iteration %d' % i) as pbar:
                for i_episode in range(int(self.num_episodes/10)):
                    st = self.env.reset()
                    # st = st['observation']
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
                        # env.render()
                        at = self.take_action(st)
                        if self.action_type == 'continuous':
                            at_con = [dis_to_con(at, self.env, self.action_dim)]
                            st1,rt1,down_,_ = self.env.step(at_con)
                        else:
                            st1,rt1,down_,_ = self.env.step(at)
                        # st1 = st1['observation']
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

    
    def plot_AC(self,return_list,episodes_len_list):
        """ 绘制 Actor Critic 算法的学习曲线 """
        episodes_list = list(range(len(return_list)))
        plt.figure(figsize=(10, 8))
        plt.suptitle('Actor Critic on {}'.format(self.env.spec.id))

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

    def run_AC(self):
        
        ''' 
            运行 Actor Critic 算法
        '''
        return_list,episodes_len_list = self.train_AC()
        self.plot_AC(return_list,episodes_len_list)

    def save_model(self, path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        

def run_trained_agent(env, agent, video_folder='Auto_Drive/highway/videos_ac'):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    video_folder = f"{video_folder}_{timestamp}"
    env = gym.wrappers.RecordVideo(env, video_folder, episode_trigger=lambda x: True)
    state = env.reset()
    # state = state['observation']
    done = False
    total_reward = 0
    while not done:
        env.render()
        action = agent.take_action(state)
        next_state, reward, done, _ = env.step(action)
        next_state = next_state['observation']
        state = next_state
        total_reward += reward
        time.sleep(0.5)
    print(f"Total Reward: {total_reward}")
    env.close()

if __name__ == "__main__":
    config = {
        "action": {
            "type": "DiscreteMetaAction"
        }
    }
    env = gym.make('highway-v0', config=config)
    print(env.observation_space)
    state_dim = env.observation_space.shape[0]
    print(state_dim)
    action_dim = env.action_space.n

    actor_lr = 1e-3
    critic_lr = 1e-2
    num_episodes = 500
    hidden_dims = [128,64,32]
    gamma = 0.98
    epsilon = 0.05
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    random.seed(9)
    np.random.seed(5)
    env.seed(3)
    torch.manual_seed(6)
    agent = ActorCritic(state_dim, action_dim, hidden_dims, actor_lr, critic_lr, 
                 gamma, device, num_episodes, env)

    agent.run_AC()

    agent.save_model('Auto_Drive/highway/AC_MODEL/actor_critic_model.pth')

    agent.load_model('Auto_Drive/highway/AC_MODEL/actor_critic_model.pth')
    run_trained_agent(env, agent)