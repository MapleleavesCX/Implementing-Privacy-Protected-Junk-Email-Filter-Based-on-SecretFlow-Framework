from secretflow.ml.boost.ss_xgb_v import Xgb
from toolfunc import timing

'''
params = {
    # for more detail, see Xgb API doc
    'num_boost_round': 5,
    'max_depth': 5,
    'learning_rate': 0.1,
    'sketch_eps': 0.08,
    'objective': 'logistic',
    'reg_lambda': 0.1,
    'subsample': 1,
    'colsample_by_tree': 1,
    'base_score': 0.5,
}
'''

class DecisionTree:
    @timing
    def __init__(self,spu):
        self.Xgb = Xgb(spu)

    @timing
    def train(self, X_train, y_train, params):
        self.model=self.Xgb.train(params,X_train, y_train)

    @timing
    def predict(self, X_test):
        return self.model.predict(X_test)


# 示例使用
if __name__ == '__main__':
    None
