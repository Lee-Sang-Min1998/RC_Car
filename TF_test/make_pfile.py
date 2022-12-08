# p file 생성
import pickle
from PIL import Image
import cv2
import numpy as np
import os

def settingValue(direction, image): # 라벨링을 위한 값 설정
    velocity = 1
    filtering_image = np.array(cv2.resize(image,(16,16)))
    filtering_image = filtering_image[:,:,0]    
    
    data = [0.0, 0, 0] # 라벨링 배열 만들기

    data[0] = direction
    data[1] = velocity
    data[2] = filtering_image

    return data

total_data = []
dir_path = "./image/" # 이미지 경로

for (root, directories, files) in os.walk(dir_path):
    for file in files:
        file_path = os.path.join(root, file)
        image = cv2.imread(file_path) # 이미지 가져오기

        cv2.imshow('disp',np.array(cv2.resize(image,dsize=(280,280), interpolation=cv2.INTER_LINEAR)))
        cv2.waitKey(1000)
        cv2.destroyWindow('disp') 

        direction = float(input("direction: "))
        settingValue(direction, image)
        total_data.append(settingValue(direction, image))



# print(np.array(filtering_image))
# print(np.array(filtering_image).shape)

with open('learning_image10.p', 'wb') as file:    # james.p 파일을 바이너리 쓰기 모드(wb)로 열기
    pickle.dump(total_data, file)