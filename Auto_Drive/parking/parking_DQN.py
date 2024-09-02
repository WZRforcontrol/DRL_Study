import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'E:\Anaconda\Projects\RL\Study'))
import random
import gym
import numpy as np
import torch
import highway_env
import collections
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import RL_algo.rl_utils as rl_utils
import warnings
warnings.filterwarnings("ignore")

def dis_to_con(discrete_action, env, action_dim):  
    action_lowbound = env.action_space.low[0]
    action_upbound = env.action_space.high[0]
    re = action_lowbound + (discrete_action /(action_dim - 1)) * (action_upbound -action_lowbound)
    return re

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)
    

class Qnet(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims) -> None:
        super(Qnet, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(state_dim, hidden_dims[0]))
        for i in range(1, len(hidden_dims)):
            self.layers.append(torch.nn.Linear(hidden_dims[i-1], hidden_dims[i]))
        self.layers.append(torch.nn.Linear(hidden_dims[-1], action_dim))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)

class DQN:
    def __init__(self,state_dim, action_dim, hidden_dims, learning_rate, 
                 gamma, epsilon, target_update, device,env) -> None:
        self.action_dim = action_dim
        self.main_q_net = Qnet(state_dim, action_dim, hidden_dims).to(device)
        self.target_q_net = Qnet(state_dim, action_dim, hidden_dims).to(device)
        self.optimizer = torch.optim.Adam(self.main_q_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
        self.device = device
        self.env = env
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.action_type = 'discrete'
        elif isinstance(self.env.action_space, gym.spaces.Box):
            self.action_type = 'continuous'
        else:
            raise ValueError("Unsupported action space type")

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.main_q_net(state).argmax().item()
        return action

    def best_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        q_value = self.main_q_net(state)
        action = torch.argmax(q_value).item()
        if self.action_type == 'continuous':
            action = [dis_to_con(action, self.env, self.action_dim)]
        return action
        
    def update(self,transition_dict):
        states = torch.FloatTensor(transition_dict['states']).to(self.device)
        actions = torch.LongTensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.FloatTensor(transition_dict['rewards']).view(-1, 1).to(self.device)
        next_states = torch.FloatTensor(transition_dict['next_states']).to(self.device)
        dones = torch.FloatTensor(transition_dict['dones']).view(-1, 1).to(self.device)

        q_values = self.main_q_net(states).gather(1, actions)
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)

        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.main_q_net.state_dict())
        self.count += 1

    def train_DQN(self,buffer_size,num_episodes,minimal_size,batch_size):
        replay_buffer = ReplayBuffer(buffer_size)
        return_list = []
        episodes_len_list = []
        for i in range(10):
            with tqdm(total=int(num_episodes/10) , desc='Iteration %d' % i) as pbar:
                for i_episode in range(int(num_episodes/10)):
                    st = self.env.reset()
                    st = st['observation']
                    done = False
                    episode_return = 0
                    episodes_len = 0
                    while not done:
                        at = self.take_action(st)
                        if self.action_type == 'continuous':
                            at_con = [dis_to_con(at, self.env, self.action_dim)]
                            st1,rt1,down_,_ = self.env.step(at_con)
                        else:
                            st1,rt1,down_,_ = self.env.step(at)
                        st1 = st1['observation']
                        replay_buffer.add(st, at, rt1, st1, down_)
                        episode_return += rt1
                        st = st1
                        done = down_
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
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                        })
                    pbar.update(1)
        return return_list,episodes_len_list

    def plot_DQN(self,return_list,episodes_len_list):
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
        return_list,episodes_len_list = self.train_DQN(buffer_size,num_episodes,minimal_size,batch_size)
        self.plot_DQN(return_list,episodes_len_list)

    def save_model(self, path):
        torch.save(self.main_q_net.state_dict(), path)

    def load_model(self, path):
        self.main_q_net.load_state_dict(torch.load(path))
        self.target_q_net.load_state_dict(torch.load(path))

def run_trained_agent(env, agent, video_folder='Auto_Drive/videos'):
    env = gym.wrappers.RecordVideo(env, video_folder, episode_trigger=lambda x: True)
    state = env.reset()
    state = state['observation']
    done = False
    total_reward = 0
    while not done:
        env.render()
        action = agent.best_action(state)
        next_state, reward, done, _ = env.step(action)
        next_state = next_state['observation']
        state = next_state
        total_reward += reward
    print(f"Total Reward: {total_reward}")
    env.close()

if __name__ == "__main__":
    config = {
        "action": {
            "type": "DiscreteMetaAction"
        }
    }
    env = gym.make('parking-v0', config=config)
    # env = gym.make('parking-v0')
    state_dim = env.observation_space['observation'].shape[0]
    action_dim = env.action_space.n
    # action_dim = 100

    lr = 1e-2
    num_episodes = 5000
    hidden_dim = [128 , 128]
    gamma = 0.98
    epsilon = 0.01
    target_update = 800
    buffer_size = 5000
    minimal_size = 1000
    batch_size = 64
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    random.seed(0)
    np.random.seed(0)
    env.seed(0)
    torch.manual_seed(0)
    agent = DQN(state_dim, action_dim, hidden_dim, lr, gamma, epsilon, target_update, device, env)

    agent.run_DQN(buffer_size,num_episodes,minimal_size,batch_size)
    agent.save_model('Auto_Drive\dqn_model.pth')

    agent.load_model('Auto_Drive\dqn_model.pth')
    run_trained_agent(env, agent)