__author__ = 'will'

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

outputs = 1

from get_image_data import *

trX,trY = get_training_data() # 훈련 데이터 가져오기
teX,teY = get_test_data() # 테스트 데이터 가져오기

seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)

model=Sequential() # 학습 모델
model.add(Dense(512, input_dim=np.shape(trX)[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(trX, trY, epochs=3, batch_size=1) # 학습

Y_prediction = model.predict(teX).flatten() # test 예측값

for i in range(len(teY)):
    label = teY[i]
    pred = Y_prediction[i]
    print("label:{:.2f}, pred:{:.2f}".format(label, pred)) # 정답과 예측값 비교


def get_direction(img): 
    print(img.shape)
#    img = np.array([np.reshape(img,img.shape**2)])
    ret =  model.predict(np.array([img])) # 이미지 예측
    return ret

# Predict direction with single image
dir=get_direction(teX[len(teX)-1]) # 이미지로 예측
print(dir[0][0])