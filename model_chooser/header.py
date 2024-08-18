'''
考虑多种模型与多种应用情况,本篇对代码标准作统一说明:
**************************************************************************************************
安全策略：

A类:MPC机器学习
隐语官方文档 
https://www.secretflow.org.cn/zh-CN/docs/secretflow/v1.5.0b0/user_guide/mpc_ml

B类:联邦学习
隐语官方文档 
https://www.secretflow.org.cn/zh-CN/docs/secretflow/v1.5.0b0/user_guide/federated_learning

**************************************************************************************************
待实现的模型：

1.逻辑回归分类器
2.朴素贝叶斯分类器
3.感知机(Perceptron)分类器
4.支持向量机分类器SVM

主要参考地址：

中文垃圾短信识别(逻辑回归分类器源代码,朴素贝叶斯分类器源代码,感知器分类器源代码) 
https://github.com/hrwhisper/SpamMessage

隐语官方文档>线性回归模型 
https://www.secretflow.org.cn/zh-CN/docs/secretflow/v1.5.0b0/user_guide/mpc_ml/linear_model

Spam Scanner(特征包含:
朴素贝叶斯分类器,
垃圾内容检测,
网络钓鱼内容检测,
可执行链接和附件检测,
病毒检测,
NSFW图像检测)
https://github.com/spamscanner/spamscanner

使用机器学习模型预测邮件是否合法(使用 Scikit-learn ML 库中的朴素贝叶斯分类器和支持向量机) 
https://github.com/abhijeet3922/Mail-Spam-Filtering


5.神经网络分类器
主要参考地址：
隐语官方文档>水平联邦:图像分类 
https://www.secretflow.org.cn/zh-CN/docs/secretflow/v1.5.0b0/tutorial


6.决策树模型分类器——垂直划分数据集
主要参考地址：
隐语官方文档>决策树模型 
https://www.secretflow.org.cn/zh-CN/docs/secretflow/v1.5.0b0/user_guide/mpc_ml/decision_tree

**************************************************************************************************
统一未处理的初始数据集格式：

第一列              第二列
判断是否为垃圾邮件   邮件文本内容

**************************************************************************************************

统一SecretFlow版本:
v1.5.0b0

**************************************************************************************************

'''

#本篇应当定义完全模型函数作为调用页

# 以下为模型选择的常量model_chose定义：
'''
LogisticRegression = 1
NaiveBayesian = 2
Perceptron = 3
SVM = 4
NN = 5
DecisionTree = 6
'''


'''
这里是主调函数页
'''


from model.SecureLogisticRegression import SecureLogisticRegression

# 其他模型的导入...

class ModelChooser:
    def __init__(self, model_id, spu, other=None):
        self.models = {
            1: SecureLogisticRegression,
            # 其他模型的映射...
        }
        self.model_class = self.models.get(model_id)
        if not self.model_class:
            raise ValueError(f"No model found for id {model_id}")
        self.model = self.model_class(spu)

    def train(self, X_train, y_train ,params):
        self.model.train(X_train, y_train ,params)

    def predict(self, X_test):
        return self.model.predict(X_test)
