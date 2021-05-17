import numpy as np
import tensorflow as tf

print('\u2022 Using TensorFlow Version:', tf.__version__)
# Add sequential model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])

xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(xs, ys, epochs=500, verbose=1)
print('==SAVE MODEL==')
saved_model_path = "./model/{}.h5".format('linear-model')

model.save(saved_model_path)
print('==MODEL SAVED==')
