# file 구조
import pickle
import numpy as np

data = pickle.load( open( "Output.p", "rb" ), encoding="latin1" ) # test 이미지 가져오기
n_images = len(data)

print(np.array(data).shape)
print(np.array(data[0]).shape)
print(np.array(data[0][0]))
print(np.array(data[0][1]))
print(np.array(data[0][2]))


# for a in data:
#     print(np.array(a[1]))

#print(np.array(data).shape)
#print(np.array(data[0]).shape)
#print(np.array(data[0][2]).shape)