import tensorflow as tf
from tensorflow.keras import layers
tf.keras.backend.set_floatx('float64')

class MLPRegressor(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim1,hidden_dim2, output_dim):
        super(MLPRegressor, self).__init__()
        self.dense1 = layers.Dense(hidden_dim1, activation='relu', dtype='float64')
        self.dense2 = layers.Dense(hidden_dim2, activation='relu', dtype='float64')
        self.dense3 = layers.Dense(output_dim, dtype='float64')
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self._loss = tf.keras.losses.mean_squared_error

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 5), dtype=tf.float64)])
    def predict(self, inputs):
        target=self(inputs)
        return  {"target": target}
        # return target

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(None, 5), dtype=tf.float64),
            tf.TensorSpec(shape=(None, 1), dtype=tf.float64),
        ])
    def train(self, inputs, targets):
        with tf.GradientTape() as tape:
            predictions = self(inputs)
            loss = self._loss(targets, predictions)
            mae = tf.keras.metrics.mean_absolute_error(targets, predictions)
        gradients = tape.gradient(loss, self.trainable_variables)
        self._optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {"loss": loss,"mae": mae}
        # return loss,mae

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(None, 5), dtype=tf.float64),
            tf.TensorSpec(shape=(None, 1), dtype=tf.float64),
        ])
    def mean_absolute_error(self, inputs, targets):
        predictions = self(inputs)
        mae = tf.keras.metrics.mean_absolute_error(targets, predictions)
        return {"mae": mae}
        # return mae

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(None, 5), dtype=tf.float64),
            tf.TensorSpec(shape=(None, 1), dtype=tf.float64),
        ])
    def mean_squared_error(self, inputs, targets):
        predictions = self(inputs)
        mse = tf.keras.metrics.mean_squared_error(targets, predictions)
        return {"mse": mse}
        # return mse


# 创建模型
model = MLPRegressor(input_dim=5, hidden_dim1=64, hidden_dim2=64,output_dim=1)

#空数据
model.train(
    tf.zeros([1, 5], dtype=tf.float64),
    tf.zeros([1, 1], dtype=tf.float64),
)

model.predict(tf.zeros((1, 5), dtype=tf.float64))

print(model.summary())

# .checkpoint保存
# checkpoint = tf.train.Checkpoint(model=model)
# checkpoint.save("check/mlp")

tf.keras.models.save_model(
    model,
    "../keras/mlp",
    signatures={
        "train": model.train,
        "predict": model.predict,
        "MAE":model.mean_absolute_error,
        "MSE":model.mean_squared_error,

    },
)



# # ________测试代码________
#
# # 构建数据集
# x_train = tf.random.normal((1000, 5), dtype='float64')
# y_train = tf.random.normal((1000, 1), dtype='float64')
# x_val = tf.random.normal((100, 5), dtype='float64')
# y_val = tf.random.normal((100, 1), dtype='float64')
#
# # 训练模型
# for i in range(100):
#     train_loss, train_mae = model.train(x_train, y_train)
#     val_mae = model.mean_absolute_error(x_val, y_val)
#     val_mse = model.mean_squared_error(x_val, y_val)
#     print("Epoch",i,"train_loss",train_loss.numpy(),"train_mae",train_mae.numpy(),"val_mae",val_mae.numpy())
#
# # 使用模型进行预测
# x_test = tf.random.normal((10, 5), dtype='float64')
# y_pred = model.predict(x_test)
# print(y_pred)