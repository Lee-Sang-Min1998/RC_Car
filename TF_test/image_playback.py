# 예측값과 실제값을 비교하기 위한 코드
__author__ = 'will'

import pickle
import cv2
import time
import numpy as np

data = pickle.load( open( "Output.p", "rb" ), encoding="latin" ) # test 이미지 가져오기
n_images = len(data)
print(n_images)
test, training = data[0:int(n_images/4)], data[int(n_images/4):]


#print (data[4200])
# print(test)

#img = data[4200][2]
#img = np.array(img,dtype=np.uint8)
#cv2.imshow('disp',np.array(cv2.resize(img,(280,280))))


cv2.namedWindow('disp')
for direcao,velocidade,img in data:
    img = np.array(img,dtype=np.uint8)
    print (direcao, velocidade) # 데이터에서 방향과 속도 추출
    cv2.imshow('disp',np.array(cv2.resize(img,(280,280))))

#    time.sleep(0.05)
    cv2.waitKey(0)
cv2.destroyAllWindows()

