# RC_CAR 구동을 위한 인터페이스


__author__ = 'will'

import numpy as np
import cv2

from picamera.array import PiRGBArray
from picamera import PiCamera

class RC_Car_Interface():

    def __init__(self):
        self.left_wheel = 0
        self.right_wheel = 0
        self.camera = PiCamera()
        self.camera.resolution = (320,320)         # set camera resolution to (320, 320)
        self.camera.color_effects = (128,128)      # set camera to black and white

    def finish_iteration(self):
        print('finish iteration')

    def set_right_speed(self, speed): # 오른쪽 속도 설정
        print('set right speed to ', speed)
    
    def set_left_speed(self, speed): # 왼쪽 속도 설정
        print('set left speed to ', speed)
        
    def get_image_from_camera(self): # 이미지 받아오기
        img = np.empty((320, 320, 3), dtype=np.uint8)
        self.camera.capture(img, 'bgr')
        
        img = img[:,:,0]           # 카메라가 흑백이므로 3차원도 같은 결과를 나타냄
                                   # 2차원 데이터 삭제
#        print(img)
        
        threshold = int(np.mean(img))*0.5 # 절반값을 임계점으로 설정 -> 흑백 구분을 위해
#        print(threshold)

        # 고정 임계값을 통하여 흑백 결정
        ret, img2 = cv2.threshold(img.astype(np.uint8), threshold, 255, cv2.THRESH_BINARY_INV)

        # 이미지 크기 재설정
        img2 = cv2.resize(img2,(16,16), interpolation=cv2.INTER_AREA )
#        cv2.imshow("Image", img2)
#        cv2.waitKey(0)
        return img2

    def stop(self):     # 로봇 정지
        print('stop')

# Test Only
# RC_Car_Interface().get_image_from_camera()