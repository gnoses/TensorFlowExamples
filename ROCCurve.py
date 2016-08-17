import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt



class ROCData():
    label = np.array([], dtype=float)
    score = np.array([], dtype=float)
    desc = ''
    # label_dense : m x 1
    # output : [m x num_classes] one hot
    def DenseToOneHot(self,labels_dense, num_classes):
        """Convert class labels from scalars to one-hot vectors."""
        labelOneHot = np.zeros((labels_dense.shape[0], num_classes))
        labelOneHot[np.arange(labels_dense.shape[0]), np.int8(labels_dense)] = 1
        return np.int8(labelOneHot)

    # label : [m x 1]
    # score : [m x classCount]
    def Add(self, label, score):
        # labelFlat = label.reshape(score.shape[2] * score.shape[3])
        # scoreFlat = score.reshape(score.shape[1], score.shape[2] * score.shape[3]).transpose(1,0)

        # labelOneHot = self.DenseToOneHot(label, classCount)
        # print 'one hot', labelOneHot.shape
        # print labelOneHot
        # print 'score ', scoreFlat.shape
        # labelOneHot : m x classCount
        # fpr, tpr, _ = roc_curve(label[:,1], score[:,1])
        if score.shape[0] == 0:
            return
        self.label = np.concatenate((self.label, label), axis=0)
        self.score = np.concatenate((self.score, score), axis=0)
        # print 'add',self.label.shape[0]
    # return tp, fp, tf, fn
    def Evaluate(self):
        tp = np.sum(np.logical_and((self.label == 1), (self.score >= 0.5)).astype(np.uint8))
        fp = np.sum(np.logical_and((self.label == 0), (self.score >= 0.5)).astype(np.uint8))
        tn = np.sum(np.logical_and((self.label == 0), (self.score < 0.5)).astype(np.uint8))
        fn = np.sum(np.logical_and((self.label == 1), (self.score < 0.5)).astype(np.uint8))
        print tp, fp, tn, fn
        return tp, fp, tn, fn

    def PlotROCCurve(self):
        self.Evaluate()

        # print 'plot',self.label.shape[0]
        plt.ioff()
        # plt.plot(self.label)
        plt.plot(self.score)
        # plt.imshow()
        fpr, tpr, _ = roc_curve(self.label, self.score)
        roc_auc = auc(fpr, tpr)

        plt.ioff()
        plt.figure(2)
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % (roc_auc))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC (%d ea)' % self.label.shape[0])
        plt.legend(loc="lower right")
        plt.show()

    def SaveData(self, filename):
        np.savez(filename, label=self.label, score=self.score)
        plt.savefig(filename + '.png')


    def LoadData(self, filename, desc):
        z = np.load(filename + '.npz')
        # print z.files
        self.label = z['label']
        self.score = z['score']
        self.desc = desc
        return self
        # print '%d data loaded' % self.label.shape[0]

class ROCCurve():
    rocList = []

    def Add(self, rocData):
        self.rocList.append(rocData)

    def Plot(self):
        for roc in self.rocList:
            fpr, tpr, _ = roc_curve(roc.label, roc.score)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=roc.desc + ' : %.03f' % (roc_auc))

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC (%d ea)' % roc.label.shape[0])
        plt.legend(loc="lower right")
        plt.show()

if __name__ == '__main__':

    rocCurve = ROCCurve()
    # for i in range(10000,40000,10000):
    if (1):
        rocData = ROCData()
        rocCurve.Add(rocData.LoadData('rocResult/ROC_val_iter_%d' % 30000, '480x360 12layer'))

    if (1):
        rocData = ROCData()
        rocCurve.Add(rocData.LoadData('rocResult/ROC_val_iter_%d' % 110000, '356x128 12layer'))

    if (1):
        rocData = ROCData()
        rocCurve.Add(rocData.LoadData('rocResult/ROC_val_iter_%d' % 100000, '480x360 vgg'))

    # for i in range(10000, 110001, 10000):
    #     rocData = ROCData()
    #     rocCurve.Add(rocData.LoadData('rocResult/ROC_val_iter_%d' % i, '%d' % i))

    rocCurve.Plot()
