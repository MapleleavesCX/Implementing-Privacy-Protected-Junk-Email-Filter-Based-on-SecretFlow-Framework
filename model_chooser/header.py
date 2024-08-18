'''
考虑多种模型与多种应用情况,本篇对代码标准作统一说明:
**************************************************************************************************
# 以下为模型选择的常量model_chose定义：

LogisticRegression = 1
DecisionTree = 2
NaiveBayesian = 3
Perceptron = 4
SVM = 5
NN = 6

**************************************************************************************************
统一未处理的初始数据集格式：

第一列              第二列
判断是否为垃圾邮件   邮件文本内容

**************************************************************************************************

统一SecretFlow版本:
v1.5.0b0

**************************************************************************************************

'''

'''
这里是主调函数页
'''
from ..other_function.toolfunc import timing

from model.SecureLogisticRegression import SecureLogisticRegression
from model.DecisionTree import DecisionTree
# 其他模型的导入...

class ModelChooser:
    @timing
    def __init__(self, model_id, spu, other=None):
        self.models = {
            1: SecureLogisticRegression,
            2: DecisionTree,
            # 其他模型的映射...
        }
        self.model_class = self.models.get(model_id)
        if not self.model_class:
            raise ValueError(f"No model found for id {model_id}")
        self.model = self.model_class(spu)

    @timing
    def train(self, X_train, y_train ,params):
        self.model.train(X_train, y_train ,params)

    @timing
    def predict(self, X_test):
        return self.model.predict(X_test)
