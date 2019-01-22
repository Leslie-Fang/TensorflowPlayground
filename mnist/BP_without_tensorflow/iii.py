import numpy as np

model = {}
model['W'] = [[1,2],2,3,4]
print model
x = [np.array(item) for item in model['W']]
print x
print x[0].shape