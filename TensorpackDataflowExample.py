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
from skimage import exposure


class CustomDataset(RNGDataFlow):
    def __init__(self, dir, targetClass):
        self.imglist = []
        lines = open(dir, 'rt').readlines()
        for line in lines:
            token = line.split(' ')
            data = token[0]
            pixellabel = token[1]
            cls = int(token[2].strip('\n'))
            if cls == targetClass:
                cls = 1
            else:
                cls = 0
            self.imglist.append((data, pixellabel, cls))

    def size(self):
        return len(self.imglist)

    def get_data(self):
        idxs = np.arange(len(self.imglist))
        #if self.shuffle:
        self.rng.shuffle(idxs)

        for k in idxs:
            fname, fname2, label = self.imglist[k]

            #fname = os.path.join(self.full_dir, fname)

            # im = cv2.imread(fname)
            im = imread(fname)
            # im = resize(im, (512,512))
            #print '!!!!', im.shape
            im = exposure.equalize_hist(im)
            im = im.reshape((1,512,512,1))

            img2 = Image.open(fname2)
            # img2 = np.array(img2.resize((512,512), Image.NEAREST))
            img2 = np.array(img2)
            img2 = img2.reshape((1,512, 512))

            assert im is not None, fname
            #if im.ndim == 2:
            #    im = np.expand_dims(im, 2).repeat(3, 2)
            label = np.array(label).reshape((1,))
            yield [im, img2,label,fname]

def Reset(ds2, process):
    ds2.reset_state()
    return PrefetchDataZMQ(ds2, process)
    # dftools.dump_dataflow_to_lmdb(ds2, 'temp.lmdb')

def Dump(ds2, filename):
    dftools.dump_dataflow_to_lmdb(ds2, filename)

def LoadLMDB(filename, process):
    ds = LMDBData(filename, shuffle=False)
    ds = LocallyShuffleData(ds, 1000)
    ds = PrefetchDataZMQ(ds, process)
    ds = LMDBDataPoint(ds)
    return ds

if __name__ == '__main__':
    #meta = ILSVRCMeta()
    # print(meta.get_synset_words_1000())

    ds = CustomDataset('/home1/gnoses/Dataset/AsanLungDisease/PixelLabel/asanStrongLabelClassification512x512/train6class.txt')
    ds2 = Reset(ds)

    for k in ds.get_data():
        print k
        break
