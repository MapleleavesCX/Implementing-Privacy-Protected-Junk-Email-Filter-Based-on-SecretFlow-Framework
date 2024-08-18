# -*- coding: utf-8 -*-
# @Date    : 2024/08/10
# @Author  : MapleleavesCX

# 这里是单节点本地模拟集群MPC-逻辑回归模型 示例

import sys
import time
import logging
import numpy as np
import pandas as pd

import ray
import spu
import secretflow as sf
from secretflow.data.split import train_test_split
from secretflow.device.driver import wait, reveal
from secretflow.data import FedNdarray, PartitionWay


from other_function.toolfunc import Divide_X, Divide_y
from model_chooser.header import ModelChooser
from other_function.TextPreprocessor import check_nltk_resources, TextPreprocessor


# 初始化 log
logging.basicConfig(stream=sys.stdout, level=logging.INFO)



# 这里是对文本进行特征工程与降维，可以跳过

#*******************************************文本处理*********************************************

'''
# 调用函数检查资源
check_nltk_resources()

# 调用文本处理,填入数据集路径名称<data_name>.csv与输出路径文件名称<new_data_name>.csv
preprocessor = TextPreprocessor()

# 加载数据
print("加载数据...")
data = preprocessor.load_data(input_file_path='SpamData.csv')

# 提取特征（内置预处理数据），可自定义提取方法method
print("预处理，并提取特征...")
preprocessor.extract_features(method='tfidf', ngram_range=(1, 1), min_df=5)

# 降维,可自设定降维方式method和压缩后的特征数量n_components
print("降维...")
preprocessor.reduce_dimensions(method='pca', n_components=1000)

# 返回输出特征矩阵和标签
print("处理完成，输出内容！")
X, y = preprocessor.get_X_y()

# # 保存处理后的数据到CSV文件
# print("保存中...")
# preprocessor.save_to_csv(output_file_path='new_SpamData.csv')
# print("over!")
'''

#*********************************读取已经完成处理的数据集**************************************

# 读取处理后的数据
print("读取处理后的数据...")
# 这里需要根据个人存放数据集的路径去更改
newdata = pd.read_csv('new_SpamData.csv')

# 提取特征和标签(会自动去除作为列名的第一行)
print('提取特征和标签...')
X = newdata.drop(columns=['Category'])
y = newdata['Category']


#****************************************标准化***********************************************

from sklearn.preprocessing import StandardScaler
# LR的训练数据集必须标准化或规范化
# 导入 StandardScaler 类，这是一个用于数据标准化的类，它会将数据转换为均值为 0，标准差为 1 的分布
scaler = StandardScaler()
# 调用 scaler.fit_transform(x) 方法对数据进行拟合和转换。
# fit 方法计算数据的均值和标准差，transform 方法使用这些统计量对数据进行标准化
X = scaler.fit_transform(X)

#*****************************************SPU设置*********************************************

ray.shutdown()


# 请使用ray 头地址.
# sf.init(parties=['alice', 'bob'], address='ray_ip:ray_port')
sf.init(parties=['alice', 'bob'], address='alice:9000')

cluster_def={
    'nodes': [
        {
            'party': 'alice',
            'address': 'alice:9100',
            'listen_addr': '0.0.0.0:9100'
        },
        {
            'party': 'bob',
            'address': 'bob:9200',
            'listen_addr': '0.0.0.0:9200'
        },
    ],
    'runtime_config': {
        'protocol': spu.spu_pb2.SEMI2K,
        'field': spu.spu_pb2.FM128,
        'sigmoid_mode': spu.spu_pb2.RuntimeConfig.SIGMOID_REAL,
    }
}

spu = sf.SPU(cluster_def=cluster_def)


#**************************************共同参数*******************************************
random_state = 1234
split_factor = 0.8

# 设置一些训练参数
params = {
    'epochs':5,
    'learning_rate':0.3,
    'batch_size':32,
    'sig_type':'t1',
    'reg_type':'logistic',
    'penalty':'l2', 
    'l2_norm':0.1
}

# 假设用户选择了模型ID为1的 SecureLogisticRegression 模型
model_id = 1


#**************************************alice 数据集划分*******************************************
# 初始化 PYU
alice = sf.PYU('alice')

print('alice 拥有前1/2的 X 部分')
X_alice_data = FedNdarray(
    partitions={
        alice: alice(Divide_X)(X, 0, 0.5)
    },
    partition_way=PartitionWay.VERTICAL,
)

# alice_y = Divide_data(y, 0.5, 1, 'r')

# # 由于label只有一列，使用Divide_data似乎总是有问题，只好为y单独写一个处理函数
# def read_y(data):
#     if isinstance(data, pd.Series):
#         data = data.to_numpy()  # 将 Series 转换为 NumPy 数组
#     return data


print('alice 拥有完全的 y 部分')
y_alice_data = FedNdarray(
    partitions={
        alice: alice(Divide_y)(y)
    },
    partition_way=PartitionWay.VERTICAL,
)

# 等待IO
wait([p.data for p in X_alice_data.partitions.values()])
wait([p.data for p in y_alice_data.partitions.values()])

print('划分训练集和测试集...')
X1_train, X1_test = train_test_split(X_alice_data, train_size=split_factor, random_state=random_state)
y1_train, y1_test = train_test_split(y_alice_data, train_size=split_factor, random_state=random_state)


#***************************************alice训练*************************************************

print('初始化...')
model_chooser = ModelChooser(model_id, spu)

# 训练模型
print('训练模型...')
model_chooser.train(X1_train, y1_train, params)


#*******************************************预测*************************************************

# 评估模型
print('评估模型...')
# 现在结果以密文保存在spu中
spu_yhat = model_chooser.predict(X1_test)
# 转换为明文
yhat = reveal(spu_yhat)

y_ = reveal(y1_test.partitions[bob])


from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

# 获得分类的曲线下面积（auc）得分
logging.info(f"auc: {roc_auc_score(y_, yhat)}")
binary_class_results = np.where(yhat > 0.5, 1, 0)
# 获得分类的准确度得分
logging.info(f"acc: {accuracy_score(y_, binary_class_results)}")
# 获取分类报告
print("classification report:")
print(classification_report(y_, binary_class_results))