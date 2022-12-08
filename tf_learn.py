# 이미지 학습

__author__ = 'will'

from keras.models import Sequential
from keras.layers import *
#from sklearn.model_selection import train_test_split

import numpy as np
#import pandas as pd
import tensorflow as tf
#import pickle
from get_image_data import *

class DNN_Driver():
    def __init__(self):
        self.trX = None
        self.trY = None
        self.teX = None
        self.teY = None
        self.model = None

    def tf_learn(self):
        self.trX, self.trY = get_training_data() # train data 가져오기
        self.teX, self.teY = get_test_data() # test data 가져오기

        seed = 0
        np.random.seed(seed)
        tf.random.set_seed(seed) # random 값

        self.model=Sequential() # 학습 모델 설정

        #print(np.array(trX).shape)
        self.trX = tf.expand_dims(self.trX, axis=-1)
        self.trX = tf.image.convert_image_dtype(self.trX, tf.float32)
        #print(np.array(trX).shape)

        # model.add(Dense(512, activation='relu'))
        self.model.add(Conv2D(16, (2, 2)))
        self.model.add(Activation('softmax'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(16, (2, 2)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())  # 이전 CNN 레이어에서 나온 3차원 배열은 1차원으로 뽑아줍니다
        self.model.add(Dense(64))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(2))

        self.model.compile(loss='mean_squared_error', optimizer='adam')

        self.model.fit(self.trX, self.trY, epochs=2, batch_size=1) # 학습
        return

    def predict_direction(self, img): # 방향 예측
        print(img.shape)
#        img = np.array([np.reshape(img,img.shape**2)])
        ret =  self.model.predict(np.array([img]))
        return ret

    def get_test_img(self): # test 이미지 받아오기
        img = self.teX[10]
        return img

        