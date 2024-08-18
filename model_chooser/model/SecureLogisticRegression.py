
# 以下为各类模型的统一定义，即定义为一个class，名称不同但是内部函数需要与header.py一致，train_data是传入的数据集,config是传入的模型参数字典
# 以逻辑回归为例：

'''
params = {
    'epochs':5,
    'learning_rate':0.3,
    'batch_size':32,
    'sig_type':'t1',
    'reg_type':'logistic',
    'penalty':'l2', 
    'l2_norm':0.1
}
'''

from secretflow.ml.linear.ss_sgd import SSRegression

class SecureLogisticRegression:
    def __init__(self,spu):
        self.model = SSRegression(spu)

    def train(self, X_train, y_train, params):
        self.model.fit(X_train, y_train, **params)

    def predict(self, X_test):
        return self.model.predict(X_test)


# 示例使用
if __name__ == '__main__':
    None
