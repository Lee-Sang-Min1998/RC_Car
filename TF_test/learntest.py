__author__ = 'will'

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.layers import Dropout

from sklearn.model_selection import train_test_split

import numpy as np
import tensorflow as tf


outputs = 1

from get_image_data import *

trX,trY = get_training_data() # 훈련 데이터 가져오기
teX,teY = get_test_data() # 테스트 데이터 가져오기

seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)

model=Sequential() # 학습 모델

print(np.array(trX).shape)
trX = tf.expand_dims(trX, axis=-1)
trX = tf.image.convert_image_dtype(trX, tf.float32)
print(np.array(trX).shape)

# model.add(Dense(512, activation='relu'))
model.add(Conv2D(16, (2, 2)))
model.add(Activation('softmax'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(16, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # 이전 CNN 레이어에서 나온 3차원 배열은 1차원으로 뽑아줍니다
model.add(Dense(64))
model.add(Dropout(0.5))
model.add(Dense(2))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(trX, trY, epochs=50, batch_size=1) # 학습

Y_prediction = model.predict(teX).flatten() # test 예측값

print("Accuracy: ", model.evaluate(teX, teY))
for i in range(len(teY)):
    label = teY[i]
    pred = Y_prediction[i]
    print("label:{:.2f}, pred:{:.2f}".format(label, pred)) # 정답과 예측값 비교


def get_direction(img): 
     # print(img.shape)
#    img = np.array([np.reshape(img,img.shape**2)])
    ret =  model.predict(np.array([img])) # 이미지 예측
    return ret

# Predict direction with single image
dir=get_direction(teX[len(teX)-1]) # 이미지로 예측
print("last image predict: ", dir[0][0])