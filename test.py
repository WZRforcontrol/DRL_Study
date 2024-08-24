import gym
import time

# 生成环境，并指定渲染模式
env = gym.make('CartPole-v1', render_mode='human')

# 环境初始化
state = env.reset()

# 循环交互
while True:
    # 渲染画面
    env.render()

    # 从动作空间随机获取一个动作
    action = env.action_space.sample()

    # agent与环境进行一步交互，并直接解包返回的五个值
    state, reward, done, truncated, info = env.step(action)
    print('state = {0}; reward = {1}'.format(state, reward))

    # 判断当前 episode 是否完成
    if done or truncated:
        print('done or truncated')
        break

    time.sleep(0.1)

# 环境结束
env.close()
