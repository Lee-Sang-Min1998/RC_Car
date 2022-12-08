# p 파일 병합

import pickle

with open('Output.p', 'rb') as file:
    input0 = pickle.load(file)

with open('learning_image20.p', 'rb') as file:
    input1 = pickle.load(file)
input0 += input1

with open('learning_image21.p', 'rb') as file:
    input2 = pickle.load(file)
input0 += input2 

with open('Output.p', 'wb') as file:
    pickle.dump(input0, file)