'''
考虑多种模型与多种应用情况,本篇对代码标准作统一说明:
**************************************************************************************************
以下为模型选择的常量 model_id 初始化输入参数 定义：

（各个模型的具体参数介绍请见对应的模型函数文件）

1: SecureLogisticRegression(spu),      逻辑回归
2: SecureDecisionTree(spu),            决策树
3: SecureNN(Server, Clients, others),  神经网络
4: ssglm(spu),                         广义线性模型

99: SecureXGboost(Server, Clients), # 不兼容当前格式的函数， 目前仅测试用， 不可调用predict

**************************************************************************************************
统一未处理的初始数据集格式：

第一列              第二列
判断是否为垃圾邮件   邮件文本内容

**************************************************************************************************

统一SecretFlow版本:
v1.5.0b0

**************************************************************************************************
'''

from ..other_function.toolfunc import timing

# 其他模型的导入...
from model.SecureLogisticRegression import SecureLogisticRegression
from model.DecisionTree import SecureDecisionTree
from model.XGboost import SecureXGboost
from model.NN import SecureNN
from model.SSGLM import ssglm


class ModelChooser:
    @timing
    def __init__(self, model_id, 
                 spu=None, 
                 Server=None, Clients=[], 
                 others=None):
        '''初始化传参内容：
        1: SecureLogisticRegression(spu),
        2: SecureDecisionTree(spu),
        3: SecureNN(Server, Clients, others),
        4: ssglm(spu),

        99:SecureXGboost(Server, Clients)
        '''
        
        self.models = {
            1: SecureLogisticRegression,
            2: SecureDecisionTree,
            3: SecureNN,
            4: ssglm,
            # 其他模型的映射...

            99: SecureXGboost,
            
        }
        self.model_class = self.models.get(model_id)
        if not self.model_class:
            raise ValueError(f"No model found for id {model_id}")
        
        self.model = self.model_class(self,  
                 spu, 
                 Server, Clients, 
                 others)

    @timing
    def train(self, X_train, y_train ,params):
        '''训练函数'''
        self.model.train(X_train, y_train ,params)

    @timing
    def predict(self, X_test):
        '''预测函数'''
        return self.model.predict(X_test)
    
    # 以下函数对部分模型不支持
    @timing
    def load_model(self, model_path):
        None
    
    @timing
    def save_model(self, model_path):
        return None
