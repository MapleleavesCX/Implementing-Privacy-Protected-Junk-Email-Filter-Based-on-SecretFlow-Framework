# Implementing-Privacy-Protected-Junk-Email-Filter-Based-on-SecretFlow-Framework
实现包含多种隐私计算机器学习模型可选的垃圾邮件过滤器，统一接口，统一格式，既方便小组成员实现与互相调用、也要方便用户的调用、还要方便其他开发者增加新的模型或者算法，即实现一个多模型、具有用户友好的、易于扩展的过滤器

secretflow版本：1.5.0b0
python:3.10

# 目录结构介绍：
data: 存放数据集
example: 快速开始一个运行样例
model_chooser: 存放模型
other_function: 其他功能函数（包括文本处理）

main_alice_MPC.py 集群模式下主节点主调函数（存在BUG，尚未修复）
main_bob_MPC.py   集群模式下从节点主调函数（存在BUG，尚未修复）
