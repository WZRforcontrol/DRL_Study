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

### RL_algo

RL_algo为经过学习并参考一些代码所编写的 DRL&RL 代码库

包含了基于模型的model_based方法和基于数据的model_free方法

model_free方法又分为了基于价值的value_based方法和基于策略的policy_based方法

**_model_based_**

- DynaQ.py : Dyna-Q
- value_policy_iteration.py : 策略迭代policy iteration, 值迭代value iteration, 截断策略迭代Truncated policy iteration

**_model_free_**

**policy_based**

- Policy_gradient.py : 

**value_based**

- TD_learning_table.py : 基于表格的时序差分方法，包含了 Sarsa, Expected Sarsa, n step Sarsa, Q learning 算法
- TD_learning_fun.py : 基于函数近似（approximate function）的时序差分方法，包括 DQN , CNN DQN , Double DQN , Dueling DQN 算法

**_env_**

- grid_world.py ： grid world 环境 : 用的赵世钰老师编写的,根据自己编写的函数更改了一些函数
- arguments.py : 环境参数

### jing

井字游戏，强化学习(第二版)中的，蛮好玩的

### reference
         
参考文献，包含了一些**经典文献**和**本项目参考的书籍**

### 其他文件

- grid_world_example.py :

### Jupyter Notebooks

本项目包含多个 Jupyter Notebook 文件，用于展示和测试不同的强化学习算法。

- DynaQ_example.ipynb : 
- policy_iteration_example1.ipynb :
- policy_iteration_example2.ipynb :
- TD_learning_fun_example.ipynb : 
- TD_learning_table_example.ipynb : 
- ten_armed_testbed.ipynb :