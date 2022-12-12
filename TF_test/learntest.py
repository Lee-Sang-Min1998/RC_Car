__author__ = 'will'

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.layers import Dropout

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import numpy as np
import tensorflow as tf


outputs = 1

from get_image_data import *

trX,trY = get_training_data() # 훈련 데이터 가져오기
teX,teY = get_test_data() # 테스트 데이터 가져오기

seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)

# 파라미터
classes = 3  # 클래스 수 결정
epochs_val = 30  # epoch 결정

# channel 차원 추가
trX = trX.reshape(trX.shape[0], 16, 16, 1)
teX = teX.reshape(teX.shape[0], 16, 16, 1)

# 원핫인코딩
enc = OneHotEncoder()
enc.fit(trY.reshape(-1, 1))
trY_onehot = enc.transform(trY.reshape(-1, 1)).toarray()
teY_onehot = enc.transform(teY.reshape(-1, 1)).toarray()

# 모델 설정
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=trX.shape[1:], activation='tanh'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation='tanh'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 학습
model.fit(trX, trY_onehot, epochs=epochs_val, batch_size=1)

# 예측
Y_prediction = model.predict(teX)

# 비교
for i in range(teX.shape[0]):
    ans = int(teY[i])
    pred_onehot = Y_prediction[i]
    pred = (np.argmax(pred_onehot, 0)-1)
    print(f"label: {ans:2d}, predict: {pred:2d}")


model.evaluate(teX, teY_onehot)

