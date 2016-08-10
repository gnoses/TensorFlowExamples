segnetRoot = '/home/gnoses/Project/SegNet'
caffe_root = segnetRoot + '/caffe-segnet/' 			# Change this to the absolute directoy to SegNet Caffe
import sys
sys.path.insert(0, caffe_root + 'python')
sys.path.insert(0, '/usr/local/cuda/lib64/')
import caffe
import lmdb
import numpy as np
from caffe.proto import caffe_pb2
import shutil
#basic setting

class LMDBTool:
    lmdb_env = None
    lbdb_txn = None
    lmdb_cursor = None
    batchsize = 0
    item_id = 0
    def __init__(self, path, batchsize, create=True):
        if (create == True):
            try:
                shutil.rmtree(path)
            except:
                pass

        self.lmdb_env = lmdb.open(path, map_size=int(1e12))
        self.lmdb_txn = self.lmdb_env.begin(write=True)
        self.lmdb_cursor = self.lmdb_txn.cursor()
        # self.datum = caffe_pb2.Datum()
        self.batchsize = batchsize
        print self.batchsize

    def Flush(self):
        # write last batch
        self.lmdb_txn.commit()
        # print 'last batch'
        # print (self.item_id + 1)

    def Put(self,data,label):
        # print self.batchsize
        datum = caffe.io.array_to_datum(data, label)
        keystr = '{:0>8d}'.format(self.item_id)
        # print keystr
        self.lmdb_txn.put(keystr, datum.SerializeToString())

        # write batch
        if (self.item_id + 1) % self.batchsize == 0:
            self.lmdb_txn.commit()
            self.lmdb_txn = self.lmdb_env.begin(write=True)
            print (self.item_id + 1)

        self.item_id += 1

    def Get(self, width, height, channel):
        data = []
        label = []
        datum = caffe_pb2.Datum()
        for key, value in self.lmdb_cursor:
            datum.ParseFromString(value)
            label.append(datum.label)
            data.append(caffe.io.datum_to_array(datum))

        data = np.array(data).reshape(-1, width, height, channel)
        label = np.array(label)
        return data, label

# test code
if __name__ == '__main__':
    count =1000
    lmdbTool = LMDBTool('data2', 1000, True)
    src = []
    for i in range(count):
        data = (np.ones([32,32,3]) * i) % 255
        src.append(data)
        label = 0
        lmdbTool.Put(data, label)
    lmdbTool.Flush()
    src = np.reshape(src, [-1,32,32,3])
    lmdbToolRead = LMDBTool('data2', 1000, False)
    data, label = lmdbToolRead.Get(32, 32, 3)
    print data.shape, src.shape
    print label.shape
    # plt.subplot(1,2,1)
    # plt.imshow(data[0,:,:,:])
    # plt.subplot(1,2,2)
    # plt.imshow(b[0,:,:,:])
    # plt.show()
    diff = np.mean((data - src) ** 2)
    print diff
