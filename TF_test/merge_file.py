# p 파일 병합

import pickle
with open('learning_image.p', 'rb') as file:
    input1 = pickle.load(file)

with open('learning_image1.p', 'rb') as file:
    input2 = pickle.load(file)

input1 += input2 

with open('Output.p', 'wb') as file:
    pickle.dump(input1, file)