# -*- coding: utf-8 -*-
# @Date    : 2024/08/23
# @Author  :Rex

from tensorflow import keras
from tensorflow.keras import layers
import secretflow as sf
from secretflow.ml.nn import FLModel
from secretflow.security.aggregation import SPUAggregator, SecureAggregator

def create_conv_model(input_shape, num_classes, name='model'):
    def create_model():
        # Create model
        model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="sigmoid"),  # 使用sigmoid适合二分类
    ])
        # Compile model
        model.compile(
            loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"]
        )
        return model

    return create_model


# __init__初始化额外参数参考列表：
'''
others = {
    'input_shape':1,
    'num_classes':(1000, ),
    'strategy':"fed_avg_w",
    'backend':"tensorflow",
}
'''
# train训练额外参数参考列表：
'''
params = {
    'validation_data':(X_test, y_test),
    'epochs':10,
    'sampler_method':"batch",
    'batch_size':128,
    'aggregate_freq':1,
}
'''


class SecureNN:

    def __init__(self,  
                 spu, 
                 Server, Clients, 
                 others):
        Input_shape = others.pop['input_shape']
        Num_classes = others.pop['num_classes']
        
        _model = create_conv_model(Input_shape, Num_classes)
        secure_aggregator = SecureAggregator(Server, Clients)

        self.model = FLModel(
            server=Server,
            device_list=Clients,
            model=_model,
            aggregator=secure_aggregator,
            **others
        )
    
    def train(self, X_train, y_train, params):
        self.model.fit(
            X_train,
            y_train,
            **params
        )

   
    def predict(self, X_test):
        return self.model.predict(X_test, batch_size=32)
'''
predictions = self.model.predict(X_test, batch_size=32)
alice_predictions = sf.reveal(predictions[alice])
bob_predictions = sf.reveal(predictions[bob])
'''

# 示例使用
if __name__ == '__main__':
    None
