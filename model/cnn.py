import tensorflow as tf

model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(
            8,
            (3, 3),
            strides=(2, 2),
            padding="valid",
            input_shape=(28, 28, 1),
            activation=tf.nn.relu,
            name="inputs",
        ),  # 14x14x8
        tf.keras.layers.Conv2D(
            16, (3, 3), strides=(2, 2), padding="valid", activation=tf.nn.relu
        ),  # 7x716
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, name="logits"),  # linear
    ]
)

tf.saved_model.save(model, "output/keras")
print("test")

