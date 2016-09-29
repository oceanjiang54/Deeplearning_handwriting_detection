import tensorflow as tf
from PIL import Image
import numpy as np
import os, sys


    
'''the size of image is 100 * 200'''

pixel = 255.0
path = "/Users/rongdilin/Desktop/cse610/Handwritten-and-data/Handprint/images"
listFileName = os.popen('find ' + path)
fileAddress = listFileName.readlines()
for i in range(2,len(fileAddress)):
    imagepath = fileAddress[i][:-1]
    img = np.array(Image.open(imagepath))
    print(img)



