"""
author:zjp
"""
import scipy.io as sio
import spectral
import tensorflow as tf
import numpy as np
from sklearn.decomposition import PCA

numpyMat = sio.loadmat("DataSet/Indian_Pines/Indian_Pines.mat")["Indian_Pines"]
labelMat = sio.loadmat("DataSet/Indian_Pines/Indian_Pines_gt.mat")["Indian_Pines_gt"]

print(numpyMat.shape)
print(labelMat.shape)
height, width, bands = numpyMat.shape
tempx = np.reshape(numpyMat, [-1, bands])
avgx = np.average(tempx, axis=0)
stdx = np.std(tempx, axis=0)
tempx = (tempx - avgx) / stdx - 1
tempx = np.reshape(tempx, [height, width, -1])
newx = np.reshape(tempx, (-1, tempx.shape[2]))
pca = PCA(n_components=30, whiten=True)
newx = pca.fit_transform(newx)
newx = np.reshape(newx, (tempx.shape[0], tempx.shape[1], 30))

numpyMat = newx[0: 144, 0: 144, :]
spectral.save_rgb("before" + '.jpg', numpyMat, (0, 3, 6))
height, width, bands = numpyMat.shape
value = tf.reshape(numpyMat, (1, height, width, bands))
value = tf.cast(value, tf.double)
kernel = tf.random.normal(shape=(3, 3, 30), dtype=tf.double)

valueErosion = tf.nn.erosion2d(value=value, filters=kernel, strides=(1, 1, 1, 1), padding="SAME",
                               dilations=(1, 1, 1, 1), data_format="NHWC")
valueErosion = tf.reshape(valueErosion, (144, 144, 30))
arrayAfter = np.array(valueErosion)

spectral.save_rgb("after" + '.jpg', arrayAfter, (0, 3, 6))
print(valueErosion.shape)
