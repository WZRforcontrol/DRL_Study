# 我的强化学习笔记 (RL & DRL)

**Author:** Z.R.Wang

**Date:** Aug 20 - ? , 2024

**Location:** Yanshan University

**Email:** [wangzhanran@stumail.ysu.edu.cn](mailto:wangzhanran@stumail.ysu.edu.cn)

## Reference

- **强化学习的数学原理** (西湖大学赵世钰老师)

  老师讲得真好，我哭死

  [Blili](https://space.bilibili.com/2044042934/channel/collectiondetail?sid=748665)

  [GitHub Repository](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning)

- **动手学强化学习**
  
  [GitHub Repository](https://github.com/boyu-ai/Hands-on-RL)

- **强化学习(第二版)**
  
  [GitHub Repository](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction)


## 文件说明

RL_algo为经过学习并参考一些代码所编写的DRL&RL代码库

包含了基于模型的model_based方法和基于数据的model_free方法

model_free方法又分为了基于价值的value_based方法和基于策略的policy_based方法

model_based

- DynaQ.py : Dyna-Q
- value_policy_iteration.py : 策略迭代policy iteration, 值迭代value iteration, 截断策略迭代Truncated policy iteration

model_free

policy_based

value_based

- TD_learning_table.py : 基于表格的时序差分方法，包含了 Sarsa, Expected Sarsa, n step Sarsa, Q learning 算法

env

- grid_world.py ： grid world 环境 : 用的赵世钰老师编写的,根据自己编写的函数更改了一些函数
- arguments.py : 环境参数