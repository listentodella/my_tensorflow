```python
import tensorflow as tf
from tensorflow import keras

# load image datas used for trainning
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
```


```python
model = keras.Sequential()
#输入层:输入数据的 shape 28x28 尺寸的图形
model.add(keras.layers.Flatten(input_shape = (28, 28)))
#中间层:128个神经元, 并指定 激活函数为 relu()
model.add(keras.layers.Dense(128, activation = tf.nn.relu))
#输出层:10个类别(神经元), 并指定激活函数为 softmax()
model.add(keras.layers.Dense(10, activation = tf.nn.softmax))
```


```python
# 模型的样子
# 28 * 28 = 784pixels
# 784pixels * 128 neurons = 100352
# (784 + 1bias) * 128 = 100480
# (128 + 1) * 10 = 1290
# model.summary()
```


```python
# train
# 使 train data 转化为 0~1 之间的数据, 有助于提高训练效果
train_images=train_images/255

# 指定 优化方法  loss_function  并显示精度
# adam 优化方法是很常用的, 尤其当输出结果是类别 类别判断
# sparse_categorical_crossentropy or categorical_crossentropy
# train_labels[0] 这种 只有整数的  train_data, 并且只有一个数据是1, 另特别称为 one-hot,类别为1
#model.compile(optimizer='adam', loss=tf.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
model.compile(optimizer=tf.optimizers.Adam(), loss=tf.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

# start trainning, 5 times
model.fit(train_images, train_labels, epochs=5)
```

    Train on 60000 samples
    Epoch 1/5
    60000/60000 [==============================] - 5s 75us/sample - loss: 0.4980 - accuracy: 0.8256
    Epoch 2/5
    60000/60000 [==============================] - 4s 69us/sample - loss: 0.3744 - accuracy: 0.8658
    Epoch 3/5
    60000/60000 [==============================] - 4s 64us/sample - loss: 0.3369 - accuracy: 0.8774
    Epoch 4/5
    60000/60000 [==============================] - 4s 71us/sample - loss: 0.3113 - accuracy: 0.8853
    Epoch 5/5
    60000/60000 [==============================] - 4s 72us/sample - loss: 0.2936 - accuracy: 0.8923
    




    <tensorflow.python.keras.callbacks.History at 0x1f703cdad48>




```python
# evaluate model
test_images_scaled=test_images/255 # 因为上面的train_images / 255 了
model.evaluate(test_images_scaled, test_labels)

# 对比train的loss & accuracy, evaluate得到的 loss & accuracy 都有所降低
# 但相差不大, 说明 train 的效果还可以
```

    10000/10000 [==============================] - ETA: 0s - loss: 0.3670 - accuracy: 0.86 - 0s 43us/sample - loss: 0.3650 - accuracy: 0.8699
    




    [0.36495736770629883, 0.8699]


