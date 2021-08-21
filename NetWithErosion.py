"""
@time: 2021/8/19 19:16
@author:zjp
将一个腐蚀模块插入到mvn中
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential


# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
# assert tf.config.experimental.get_memory_growth(physical_devices[0]) == True


class Net(keras.Model):

    def __init__(self, class_num=16):
        super(Net, self).__init__()
        self.conv3d_1 = keras.layers.Conv3D(filters=1, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='valid',
                                            data_format="channels_first", activation=tf.nn.relu)
        self.conv3d_2 = keras.layers.Conv3D(filters=2, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same',
                                            data_format="channels_first", dilation_rate=(3, 3, 1),
                                            activation=tf.nn.relu)
        self.conv3d_3 = keras.layers.Conv3D(filters=2, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same',
                                            data_format="channels_first", dilation_rate=(5, 5, 1),
                                            activation=tf.nn.relu)
        self.conv3d_4 = keras.layers.Conv3D(filters=4, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='valid',
                                            data_format="channels_first", dilation_rate=(1, 1, 1),
                                            activation=tf.nn.relu)
        self.conv3d_5 = keras.layers.Conv3D(filters=4, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='valid',
                                            data_format="channels_first", dilation_rate=(1, 1, 1),
                                            activation=tf.nn.relu)
        self.conv3d_6 = keras.layers.Conv3D(filters=8, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='valid',
                                            data_format="channels_first", dilation_rate=(1, 1, 1),
                                            activation=tf.nn.relu)
        self.flat = keras.layers.Flatten(data_format="channels_first")
        self.reshape1 = keras.layers.Reshape((9, 9, 30))
        self.reshape2 = keras.layers.Reshape((1, 9, 9, 30))
        self.fc2 = keras.layers.Dense(units=class_num, activation=tf.nn.softmax)

    def call(self, inputs):
        out = self.reshape1(inputs)
        out = tf.cast(out, tf.double)
        # print(out.shape)
        kernel = tf.random.normal(shape=(3, 3, 30), dtype=tf.double)
        out = tf.nn.erosion2d(value=out, filters=kernel, strides=(1, 1, 1, 1), padding="SAME",
                              dilations=(1, 1, 1, 1), data_format="NHWC")
        out = tf.nn.dilation2d(out, filters=kernel, strides=(1, 1, 1, 1), padding="SAME",
                              dilations=(1, 1, 1, 1), data_format="NHWC")
        # print(out.shape)
        out = self.reshape2(out)
        print(out.shape)
        out = self.conv3d_1(inputs)
        out_near = self.conv3d_2(out)
        out_far = self.conv3d_3(out_near)
        out = layers.add([out_near, out_far])
        out = self.conv3d_4(out)
        out = self.conv3d_5(out)
        out = self.conv3d_6(out)
        out = self.flat(out)
        out = self.fc2(out)
        return out


def create_model(class_num=16):
    return Net(class_num=class_num)


if __name__ == "__main__":
    net_test = Net(16)
    input = tf.random.normal((500, 1, 9, 9, 30))
    out = net_test(input)
    print(out.shape)
    print(net_test.summary())
