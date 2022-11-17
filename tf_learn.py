# 이미지 학습

__author__ = 'will'

from keras.models import Sequential
from keras.layers import Dense
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
        self.model.add(Dense(512, input_dim=np.shape(self.trX)[1], activation='relu')) # 입력층
        self.model.add(Dense(64, activation='relu')) # 은닉층
        self.model.add(Dense(1)) # 출력층

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

        