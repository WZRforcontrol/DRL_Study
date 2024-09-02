"""
    Actor Critic method
    

    代码库中包括 Actor Critic TRPO TRPO Continuous  PPO PPO Continuous算法的实现

    Author: Z.R.Wang
    Aug,29,2024 in YSU
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
import copy


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
    

class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims):
        super(PolicyNetContinuous, self).__init__()
        self.hidden_layers = torch.nn.ModuleList()
        
        # 输入层到第一个隐藏层
        self.hidden_layers.append(torch.nn.Linear(state_dim, hidden_dims[0]))
        
        # 隐藏层之间
        for i in range(1, len(hidden_dims)):
            self.hidden_layers.append(torch.nn.Linear(hidden_dims[i-1], hidden_dims[i]))
        
        # 最后一个隐藏层到输出层
        self.fc_mu = torch.nn.Linear(hidden_dims[-1], action_dim)
        self.fc_std = torch.nn.Linear(hidden_dims[-1], action_dim)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        mu = 2.0 * torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x))
        return mu, std  # 高斯分布的均值和标准差
        

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
        # state = torch.FloatTensor(np.array(state)).to(self.device)  # 再转换为张量
        state = torch.FloatTensor([state]).to(self.device)  # 再转换为张量
        probs = self.actor(state) # 得到动作概率
        action_dist = torch.distributions.Categorical(probs) # 生成一个分布, 以便采样, 也可以用torch.multinomial(probs, 1)来采样
        action = action_dist.sample() # 从分布中采样一个动作
        return action.item() # 返回采样的动作
    
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
        
    
class TRPO:
    '''
        TRPO 算法
        在使用 TRPO（Trust Region Policy Optimization）算法时，调节参数是优化算法性能的关键。以下是对 gamma, lmbda, kl_constraint, alpha 这几个参数的详细解释和调节建议：

        1. gamma (折扣因子)
        解释: gamma 是折扣因子，用于平衡当前奖励和未来奖励的权重。值越接近 1，智能体越重视未来的奖励。
        调节建议:
        一般取值范围在 [0.9, 0.99]。
        如果任务需要智能体更关注长期回报，可以将 gamma 调高。
        如果任务需要智能体更关注短期回报，可以将 gamma 调低。
        2. lmbda (GAE 参数)
        解释: lmbda 是广义优势估计（GAE, Generalized Advantage Estimation）的参数，用于平滑优势函数的估计。
        调节建议:
        一般取值范围在 [0.9, 1.0]。
        较高的 lmbda 值（接近 1）会使优势估计更加平滑，但可能会引入偏差。
        较低的 lmbda 值会使优势估计更加准确，但可能会增加方差。
        3. kl_constraint (KL 距离最大限制)
        解释: kl_constraint 是 KL 散度的最大限制，用于控制策略更新的步长，防止策略更新过大。
        调节建议:
        一般取值范围在 [0.001, 0.01]。
        较小的 kl_constraint 值会使策略更新更加保守，减少策略崩溃的风险，但可能会减慢学习速度。
        较大的 kl_constraint 值会使策略更新更加激进，加快学习速度，但可能会增加策略崩溃的风险。
        4. alpha (线性搜索参数)
        解释: alpha 是线性搜索的参数，用于控制线性搜索的步长。
        调节建议:
        一般取值范围在 [0.1, 0.9]。
        较小的 alpha 值会使线性搜索更加保守，减少策略崩溃的风险，但可能会减慢学习速度。
        较大的 alpha 值会使线性搜索更加激进，加快学习速度，但可能会增加策略崩溃的风险。
        调节策略
        初始设置:

        gamma = 0.98
        lmbda = 0.95
        kl_constraint = 0.005
        alpha = 0.5
        逐步调节:

        固定其他参数，调节一个参数: 每次只调节一个参数，观察其对算法性能的影响。
        使用网格搜索: 可以使用网格搜索的方法，尝试不同参数组合，找到最优参数。
        监控性能指标: 通过监控总回报、收敛速度等指标，评估参数调节的效果。
    '''
    def __init__(self, state_dim, action_dim, hidden_dims, critic_lr, 
                 gamma, lmbda, kl_constraint, alpha, device, num_episodes, env) -> None:
        # 策略网络
        self.actor = PolicyNet(state_dim, action_dim, hidden_dims).to(device) 
        # 价值网络
        self.critic = ValueNet(state_dim, hidden_dims).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr) # 价值网络优化器
        self.gamma = gamma
        self.lmbda = lmbda  # GAE参数
        self.kl_constraint = kl_constraint  # KL距离最大限制
        self.alpha = alpha  # 线性搜索参数
        self.device = device
        self.num_episodes = num_episodes
        self.env = env 
        self.action_dim = action_dim

    def take_action(self, state):
        state = torch.FloatTensor([state]).to(self.device)  # 再转换为张量
        probs = self.actor(state) # 得到动作概率
        action_dist = torch.distributions.Categorical(probs) # 生成一个分布, 以便采样, 也可以用torch.multinomial(probs, 1)来采样
        action = action_dist.sample() # 从分布中采样一个动作
        return action.item() # 返回采样的动作

    def hessian_matrix_vector_product(self, states, old_action_dists, vector):
        # 计算黑塞矩阵和一个向量的乘积
        new_action_dists = torch.distributions.Categorical(self.actor(states))
        kl = torch.mean(
            torch.distributions.kl.kl_divergence(old_action_dists,
                                                 new_action_dists))  # 计算平均KL距离
        kl_grad = torch.autograd.grad(kl,
                                      self.actor.parameters(),
                                      create_graph=True)
        kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])
        # KL距离的梯度先和向量进行点积运算
        kl_grad_vector_product = torch.dot(kl_grad_vector, vector)
        grad2 = torch.autograd.grad(kl_grad_vector_product,
                                    self.actor.parameters())
        grad2_vector = torch.cat([grad.view(-1) for grad in grad2])
        return grad2_vector

    def conjugate_gradient(self, grad, states, old_action_dists):  # 共轭梯度法求解方程
        x = torch.zeros_like(grad)
        r = grad.clone()
        p = grad.clone()
        rdotr = torch.dot(r, r)
        for i in range(10):  # 共轭梯度主循环
            Hp = self.hessian_matrix_vector_product(states, old_action_dists,
                                                    p)
            alpha = rdotr / torch.dot(p, Hp)
            x += alpha * p
            r -= alpha * Hp
            new_rdotr = torch.dot(r, r)
            if new_rdotr < 1e-10:
                break
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
        return x

    def compute_surrogate_obj(self, states, actions, advantage, old_log_probs,actor):  # 计算策略目标
        log_probs = torch.log(actor(states).gather(1, actions))
        ratio = torch.exp(log_probs - old_log_probs)
        return torch.mean(ratio * advantage)

    def line_search(self, states, actions, advantage, old_log_probs,
                    old_action_dists, max_vec):  # 线性搜索
        old_para = torch.nn.utils.convert_parameters.parameters_to_vector(
            self.actor.parameters())
        old_obj = self.compute_surrogate_obj(states, actions, advantage,
                                             old_log_probs, self.actor)
        for i in range(15):  # 线性搜索主循环
            coef = self.alpha**i
            new_para = old_para + coef * max_vec
            new_actor = copy.deepcopy(self.actor)
            torch.nn.utils.convert_parameters.vector_to_parameters(
                new_para, new_actor.parameters())
            new_action_dists = torch.distributions.Categorical(
                new_actor(states))
            kl_div = torch.mean(
                torch.distributions.kl.kl_divergence(old_action_dists,
                                                     new_action_dists))
            new_obj = self.compute_surrogate_obj(states, actions, advantage,
                                                 old_log_probs, new_actor)
            if new_obj > old_obj and kl_div < self.kl_constraint:
                return new_para
        return old_para

    def policy_learn(self, states, actions, old_action_dists, old_log_probs,advantage):  # 更新策略函数
        surrogate_obj = self.compute_surrogate_obj(states, actions, advantage,
                                                   old_log_probs, self.actor)
        grads = torch.autograd.grad(surrogate_obj, self.actor.parameters())
        obj_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
        # 用共轭梯度法计算x = H^(-1)g
        descent_direction = self.conjugate_gradient(obj_grad, states,
                                                    old_action_dists)

        Hd = self.hessian_matrix_vector_product(states, old_action_dists,
                                                descent_direction)
        max_coef = torch.sqrt(2 * self.kl_constraint /
                              (torch.dot(descent_direction, Hd) + 1e-8))
        new_para = self.line_search(states, actions, advantage, old_log_probs,
                                    old_action_dists,
                                    descent_direction * max_coef)  # 线性搜索
        torch.nn.utils.convert_parameters.vector_to_parameters(
            new_para, self.actor.parameters())  # 用线性搜索后的参数更新策略
        
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
        
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                      td_error.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1,
                                                            actions)).detach()
        old_action_dists = torch.distributions.Categorical(
            self.actor(states).detach())
        critic_loss = torch.mean(
            F.mse_loss(self.critic(states), td_target.detach()))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()  # 更新价值函数
        # 更新策略函数
        self.policy_learn(states, actions, old_action_dists, old_log_probs,
                          advantage)

    def train_TRPO(self):
        """ 
            运行 TRPO 算法 
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

    
    def plot_TRPO(self,return_list,episodes_len_list):
        """ 绘制 TRPO 算法的学习曲线 """
        episodes_list = list(range(len(return_list)))
        plt.figure(figsize=(10, 8))
        plt.suptitle('TRPO on {}'.format(self.env.spec.id))

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

    def run_TRPO(self):
        
        ''' 
            运行 TRPO 算法
        '''
        return_list,episodes_len_list = self.train_TRPO()
        self.plot_TRPO(return_list,episodes_len_list)
        

class TRPOContinuous:
    '''
        TRPOContinuous 算法
    '''
    def __init__(self, state_dim, action_dim, hidden_dims, critic_lr, 
                 gamma, lmbda, kl_constraint, alpha, device, num_episodes, env) -> None:
        # 策略网络
        self.actor = PolicyNetContinuous(state_dim, action_dim, hidden_dims).to(device) 
        # 价值网络
        self.critic = ValueNet(state_dim, hidden_dims).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr) # 价值网络优化器
        self.gamma = gamma
        self.lmbda = lmbda  # GAE参数
        self.kl_constraint = kl_constraint  # KL距离最大限制
        self.alpha = alpha  # 线性搜索参数
        self.device = device
        self.num_episodes = num_episodes
        self.env = env 
        self.action_dim = action_dim

    def take_action(self, state):
        state = torch.FloatTensor([state]).to(self.device)  # 再转换为张量
        mu, std = self.actor(state) # 得到动作概率
        action_dist = torch.distributions.Normal(mu, std) # 生成一个分布, 以便采样, 也可以用torch.multinomial(probs, 1)来采样
        action = action_dist.sample() # 从分布中采样一个动作
        return [action.item()] # 返回采样的动作

    def hessian_matrix_vector_product(self,
                                      states,
                                      old_action_dists,
                                      vector,
                                      damping=0.1):
        mu, std = self.actor(states)
        new_action_dists = torch.distributions.Normal(mu, std)
        kl = torch.mean(
            torch.distributions.kl.kl_divergence(old_action_dists,
                                                 new_action_dists))
        kl_grad = torch.autograd.grad(kl,
                                      self.actor.parameters(),
                                      create_graph=True)
        kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])
        kl_grad_vector_product = torch.dot(kl_grad_vector, vector)
        grad2 = torch.autograd.grad(kl_grad_vector_product,
                                    self.actor.parameters())
        grad2_vector = torch.cat(
            [grad.contiguous().view(-1) for grad in grad2])
        return grad2_vector + damping * vector

    def conjugate_gradient(self, grad, states, old_action_dists):
        x = torch.zeros_like(grad)
        r = grad.clone()
        p = grad.clone()
        rdotr = torch.dot(r, r)
        for i in range(10):
            Hp = self.hessian_matrix_vector_product(states, old_action_dists,
                                                    p)
            alpha = rdotr / torch.dot(p, Hp)
            x += alpha * p
            r -= alpha * Hp
            new_rdotr = torch.dot(r, r)
            if new_rdotr < 1e-10:
                break
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
        return x

    def compute_surrogate_obj(self, states, actions, advantage, old_log_probs,
                              actor):
        mu, std = actor(states)
        action_dists = torch.distributions.Normal(mu, std)
        log_probs = action_dists.log_prob(actions)
        ratio = torch.exp(log_probs - old_log_probs)
        return torch.mean(ratio * advantage)

    def line_search(self, states, actions, advantage, old_log_probs,
                    old_action_dists, max_vec):
        old_para = torch.nn.utils.convert_parameters.parameters_to_vector(
            self.actor.parameters())
        old_obj = self.compute_surrogate_obj(states, actions, advantage,
                                             old_log_probs, self.actor)
        for i in range(15):
            coef = self.alpha**i
            new_para = old_para + coef * max_vec
            new_actor = copy.deepcopy(self.actor)
            torch.nn.utils.convert_parameters.vector_to_parameters(
                new_para, new_actor.parameters())
            mu, std = new_actor(states)
            new_action_dists = torch.distributions.Normal(mu, std)
            kl_div = torch.mean(
                torch.distributions.kl.kl_divergence(old_action_dists,
                                                     new_action_dists))
            new_obj = self.compute_surrogate_obj(states, actions, advantage,
                                                 old_log_probs, new_actor)
            if new_obj > old_obj and kl_div < self.kl_constraint:
                return new_para
        return old_para

    def policy_learn(self, states, actions, old_action_dists, old_log_probs,
                     advantage):
        surrogate_obj = self.compute_surrogate_obj(states, actions, advantage,
                                                   old_log_probs, self.actor)
        grads = torch.autograd.grad(surrogate_obj, self.actor.parameters())
        obj_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
        descent_direction = self.conjugate_gradient(obj_grad, states,
                                                    old_action_dists)
        Hd = self.hessian_matrix_vector_product(states, old_action_dists,
                                                descent_direction)
        max_coef = torch.sqrt(2 * self.kl_constraint /
                              (torch.dot(descent_direction, Hd) + 1e-8))
        new_para = self.line_search(states, actions, advantage, old_log_probs,
                                    old_action_dists,
                                    descent_direction * max_coef)
        torch.nn.utils.convert_parameters.vector_to_parameters(
            new_para, self.actor.parameters())

    def update(self, transition_dict):
        # 将数据转换为张量
        states = torch.FloatTensor(transition_dict['states']).to(self.device)
        actions = torch.FloatTensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.FloatTensor(transition_dict['rewards']).view(-1, 1).to(self.device)
        next_states = torch.FloatTensor(transition_dict['next_states']).to(self.device)
        dones = torch.FloatTensor(transition_dict['dones']).view(-1, 1).to(self.device)
        
        rewards = (rewards + 8.0) / 8.0  # 对奖励进行修改,方便训练

        #  
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones) # 计算TD目标
        td_error = td_target - self.critic(states) # 计算TD误差
        
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                      td_error.cpu()).to(self.device)
        mu, std = self.actor(states) # 得到均值和标准差
        old_action_dists = torch.distributions.Normal(mu.detach(),
                                                      std.detach()) # 旧的动作分布
        old_log_probs = old_action_dists.log_prob(actions) # 旧的动作对数概率
        critic_loss = torch.mean(
            F.mse_loss(self.critic(states), td_target.detach())) # 价值网络损失
        self.critic_optimizer.zero_grad() # 优化器梯度清零
        critic_loss.backward() # 反向传播
        self.critic_optimizer.step() # 更新价值函数
        self.policy_learn(states, actions, old_action_dists, old_log_probs,
                          advantage) # 更新策略函数

    def train_TRPOcon(self):
        """ 
            运行 TRPOContinuous 算法 
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

    
    def plot_TRPOcon(self,return_list,episodes_len_list):
        """ 绘制 TRPOContinuous 算法的学习曲线 """
        episodes_list = list(range(len(return_list)))
        plt.figure(figsize=(10, 8))
        plt.suptitle('TRPOContinuous on {}'.format(self.env.spec.id))

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

    def run_TRPOcon(self):
        
        ''' 
            运行 TRPOContinuous 算法
        '''
        return_list,episodes_len_list = self.train_TRPOcon()
        self.plot_TRPOcon(return_list,episodes_len_list)       
        

class PPO:
    '''
        PPO 算法,采用截断方式
    '''
    def __init__(self, state_dim, action_dim, hidden_dims, actor_lr, critic_lr, 
                 lmbda, epochs, eps, gamma, device, num_episodes, env) -> None:
        # 策略网络
        self.actor = PolicyNet(state_dim, action_dim, hidden_dims).to(device) 
        # 价值网络
        self.critic = ValueNet(state_dim, hidden_dims).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr) # 策略网络优化器
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr) # 价值网络优化器
        self.gamma = gamma
        self.lmbda = lmbda  # GAE参数
        self.eps = eps  # PPO中截断范围的参数
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.device = device
        self.num_episodes = num_episodes
        self.env = env 
        self.action_dim = action_dim

    def take_action(self, state):
        state = torch.FloatTensor([state]).to(self.device)  # 再转换为张量
        probs = self.actor(state) # 得到动作概率
        action_dist = torch.distributions.Categorical(probs) # 生成一个分布, 以便采样, 也可以用torch.multinomial(probs, 1)来采样
        action = action_dist.sample() # 从分布中采样一个动作
        return action.item() # 返回采样的动作
        
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
        
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                               td_error.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1,
                                                            actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage  # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

    def train_PPO(self):
        """ 
            运行 PPO 算法 
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

    
    def plot_PPO(self,return_list,episodes_len_list):
        """ 绘制 PPO 算法的学习曲线 """
        episodes_list = list(range(len(return_list)))
        plt.figure(figsize=(10, 8))
        plt.suptitle('PPO on {}'.format(self.env.spec.id))

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

    def run_PPO(self):
        
        ''' 
            运行 PPO 算法
        '''
        return_list,episodes_len_list = self.train_PPO()
        self.plot_PPO(return_list,episodes_len_list)
     

class PPOContinuous:
    '''
        处理连续动作的 PPOContinuous 算法,采用截断方式
    '''
    def __init__(self, state_dim, action_dim, hidden_dims, actor_lr, critic_lr, 
                 lmbda, epochs, eps, gamma, device, num_episodes, env) -> None:
        # 策略网络
        self.actor = PolicyNetContinuous(state_dim, action_dim, hidden_dims).to(device) 
        # 价值网络
        self.critic = ValueNet(state_dim, hidden_dims).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr) # 策略网络优化器
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr) # 价值网络优化器
        self.gamma = gamma
        self.lmbda = lmbda  # GAE参数
        self.eps = eps  # PPO中截断范围的参数
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.device = device
        self.num_episodes = num_episodes
        self.env = env 
        self.action_dim = action_dim

    def take_action(self, state):
        state = torch.FloatTensor([state]).to(self.device)  # 再转换为张量
        mu, sigma = self.actor(state) # 得到动作概率
        action_dist = torch.distributions.Normal(mu, sigma) # 生成一个分布, 以便采样, 也可以用torch.multinomial(probs, 1)来采样
        action = action_dist.sample() # 从分布中采样一个动作
        return [action.item()] # 返回采样的动作
        
    def update(self, transition_dict):
        # 将数据转换为张量
        states = torch.FloatTensor(transition_dict['states']).to(self.device)
        actions = torch.FloatTensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.FloatTensor(transition_dict['rewards']).view(-1, 1).to(self.device)
        next_states = torch.FloatTensor(transition_dict['next_states']).to(self.device)
        dones = torch.FloatTensor(transition_dict['dones']).view(-1, 1).to(self.device)
        
        rewards = (rewards + 8.0) / 8.0  # 和TRPO一样,对奖励进行修改,方便训练

        #  
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones) # 计算TD目标
        td_error = td_target - self.critic(states) # 计算TD误差
        
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                               td_error.cpu()).to(self.device)
        mu, std = self.actor(states)
        action_dists = torch.distributions.Normal(mu.detach(), std.detach())
        # 动作是正态分布
        old_log_probs = action_dists.log_prob(actions)

        for _ in range(self.epochs):
            mu, std = self.actor(states)
            action_dists = torch.distributions.Normal(mu, std)
            log_probs = action_dists.log_prob(actions)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

    def train_PPOcon(self):
        """ 
            运行 PPOContinuous 算法 
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

    
    def plot_PPOcon(self,return_list,episodes_len_list):
        """ 绘制 PPOContinuous 算法的学习曲线 """
        episodes_list = list(range(len(return_list)))
        plt.figure(figsize=(10, 8))
        plt.suptitle('PPO Continuous on {}'.format(self.env.spec.id))

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

    def run_PPOcon(self):
        
        ''' 
            运行 PPOContinuous 算法
        '''
        return_list,episodes_len_list = self.train_PPOcon()
        self.plot_PPOcon(return_list,episodes_len_list)








