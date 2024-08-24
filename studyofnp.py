import numpy as np

print(np.arange(10))
# 创建一个示例数组
arr = np.array([[1, 2, 3],
                [4, 5, 6]])

# 在整个数组中查找最大值的索引
max_index = np.argmax(arr)
print("最大值的索引（扁平化数组中的位置）：", max_index)  # 输出: 5 (数组扁平化后，6 是第 5 个元素)

# 指定沿着某个轴（例如列）查找最大值的索引
max_index_along_axis0 = np.argmax(arr, axis=0)
print("沿着轴0查找最大值的索引：", max_index_along_axis0)  # 输出: [1 1 1] (每列的最大值索引分别为第1行)

max_index_along_axis1 = np.argmax(arr, axis=1)
print("沿着轴1查找最大值的索引：", max_index_along_axis1)  # 输出: [2 2] (每行的最大值索引分别为第2列)



# 生成一个形状为 (3, 4) 的随机数数组
arr = np.random.rand(3, 4)
print("随机数数组：")
print(arr)

# 生成一个形状为 (2, 3, 2) 的随机数数组
arr2 = np.random.rand(2, 3, 2)
print("\n随机数数组（3维）：")
print(arr2)


# 示例数组
indices = np.array([10, 20, 30, 40, 50])

# 从数组中随机选择一个元素
choice = np.random.choice(indices)
print("随机选择的元素：", choice)

# 从数组中随机选择三个元素（有放回抽样）
choices = np.random.choice(indices, size=3)
print("随机选择的三个元素：", choices)

# 从数组中随机选择两个元素，并且不允许重复抽样（无放回抽样）
choices_no_replace = np.random.choice(indices, size=2, replace=False)
print("无放回抽样的两个元素：", choices_no_replace)

# 指定抽取每个元素的概率
probabilities = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
choices_with_prob = np.random.choice(indices, size=4, p=probabilities)
print("根据指定概率进行抽样的四个元素：", choices_with_prob)
