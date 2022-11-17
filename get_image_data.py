# 이미지 학습을 위한 train, test data 만들기


__author__ = 'will'

import pickle
import numpy as np

data = pickle.load( open( "trainingdata.p", "rb" ), encoding="latin1" ) # 이미지 불러오기
n_images = len(data)
test, training = data[0:int(n_images/3)], data[int(n_images/3):] # 2:1 비율로 훈련, 테스트 데이터 나누기

def get_training_data():

    trX = np.array([np.reshape(a[2],a[2].shape[0]**2) for a in training]) # train x축 데이터 수집
    print(np.shape(trX)[1])
    trY = np.zeros((len(training)),dtype=np.float) # train y축 데이터 수집
    for i, data in enumerate(training):
        trY[i] = float(data[0])
    return trX, trY

def get_test_data():
    teX = np.array([np.reshape(a[2],a[2].shape[0]**2) for a in test]) # test x축 데이터 수집
    teY = np.zeros((len(test)),dtype=np.float) # test x축 데이터 수집
    for i, data in enumerate(test):
        teY[i] = float(data[0])
    return teX,teY

