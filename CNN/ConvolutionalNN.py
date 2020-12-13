from MultiLayerPerceptron import *
from CNN import *
import mnist_loader
import matplotlib.pyplot as plt
import numpy as np
training, validation, test = mnist_loader.load_data()
offset = 1000
t = training[0][0:offset],training[1][0:offset]
offsetv = 500
v = validation[0][0:offsetv],validation[1][0:offsetv]



#reshape x 28:28
#ndimage.convolve dla kazdego filtra i zapisac wynik w jakiejs liscie


a = CNN([980,50,10],None,3,1,1)



a.train(t,v,20   ,0.001,64,softplus_function,sigmoid_function,20)
