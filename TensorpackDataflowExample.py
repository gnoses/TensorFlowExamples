import os
import six
import numpy as np
import cv2
from tensorpack.dataflow import RNGDataFlow
from skimage.io import imread
from skimage.transform import resize
from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
import PIL.Image as Image


__all__ = ['CustomDataset']


class CustomDataset(RNGDataFlow):
    def __init__(self, dir):
        self.imglist = []
        lines = open(dir, 'rt').readlines()
        for line in lines:
            token = line.split(' ')
            data = token[0]
            pixellabel = token[1]
            cls = int(token[2].strip('\n'))
            if cls > 0:
                cls = 1
            self.imglist.append((data, pixellabel, cls))

    def size(self):
        return len(self.imglist)

    def get_data(self):
        idxs = np.arange(len(self.imglist))
        #if self.shuffle:
        self.rng.shuffle(idxs)
        for k in idxs:
            fname, fname2, label = self.imglist[k]    
            im = imread(fname)    
            img2 = Image.open(fname2)    
            
            yield [im, img2, label, fname]

    def Reset(self):
        self.reset_state()
        return PrefetchDataZMQ(self, 4)

def Process(epoch):
    print('#epoch : %d' % (epoch)),
    generator = ds.get_data()
    for i, k in enumerate(generator):
        # print i, k[3]
        pass
    print i

if __name__ == '__main__':
    
    ds = CustomDataset('yourdatasetpath/train.txt')
    ds.reset_state()
    size = ds.size()
        for epoch in range(10):
        Process(epoch)
