import tensorflow as tf
import os
from tensorflow.python.framework import graph_io
import numpy as np

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train,  epochs = 5)
model.evaluate(x_test, y_test)

var_sizes = [np.product(list(map(int, v.shape))) * v.dtype.size
             for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)]
print(sum(var_sizes) / (1024 ** 2), 'MB')

model.summary()

'''filepath='./my_model'
tf.keras.models.save_model(
    model,
    filepath,
    overwrite=True,
    include_optimizer=True
)
'''
'''
sess = tf.keras.backend.get_session()
saver = tf.train.Saver()
save_path = saver.save(sess, "./model.ckpt")
print("Model saved in file: %s" % save_path)


frozen = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, [node.op.name for node in model.outputs])
graph_io.write_graph(frozen, './', 'inference_graph.pb', as_text=False)
'''
