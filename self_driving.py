# 이미지 예측 및 방향 설정

__author__ = 'will'

from rc_car_interface import RC_Car_Interface
from tf_learn import DNN_Driver
import numpy as np
import time
import cv2

class SelfDriving:

    def __init__(self):
        self.rc_car_cntl = RC_Car_Interface() # 인터페이스 가져옴
        self.dnn_driver = DNN_Driver() 

        self.rc_car_cntl.set_left_speed(0) # 왼쪽 속도 0
        self.rc_car_cntl.set_right_speed(0) # 오른쪽 속도 0
    
        self.velocity = 0 # 속도 0
        self.direction = 0 # 방향 0
    
        self.dnn_driver.tf_learn() #
    
    def rc_car_control(self, direction): # 방향에 따른 속도 조절
        #calculate left and right wheel speed with direction
        if direction < -1.0:
            direction = -1.0
        if direction > 1.0:
            direction = 1.0
        if direction < 0.0: # 왼쪽 방향이면 왼쪽 속도 증가
            left_speed = 1.0+direction
            right_speed = 1.0
        else: # 오른쪽 방향이면 오른쪽 속도 증가
            right_speed = 1.0-direction
            left_speed = 1.0

        self.rc_car_cntl.set_right_speed(right_speed) # 오른쪽 속도 설정
        self.rc_car_cntl.set_left_speed(left_speed) # 왼쪽 속도 설정

    def drive(self):
        while True:

# For test only, get image from DNN test images
#            img from get_test_img() returns [256] array. Do not call np.reshape()
#            img = self.dnn_driver.get_test_img()
#            direction = self.dnn_driver.predict_direction(img)

            img = self.rc_car_cntl.get_image_from_camera() # 이미지 받아오기
# predict_direction wants [256] array, not [16,16]. Thus call np.reshape to convert [16,16] to [256] array
            #img = np.reshape(img,img.shape[0]**2)

            direction = self.dnn_driver.predict_direction(img)         # 이미지를 통한 방향 예측
            print(direction[0][0])
            self.rc_car_control(direction[0][0]) # 방향/속도 설정

            # For debugging, show image
#            cv2.imshow("target",  cv2.resize(img, (280, 280)) )
#            cv2.waitKey(0)

            time.sleep(0.001)

        self.rc_car_cntl.stop()
        cv2.destroyAllWindows()

SelfDriving().drive()