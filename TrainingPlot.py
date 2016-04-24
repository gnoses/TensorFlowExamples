import matplotlib
import matplotlib.pyplot as plt
import glob
import os
import time
import numpy as np

class StopWatch():
    startTime = time.time()
    def __init__(self):
        self.startTime = time.time()
    def StartTime(self):
        self.startTime = time.time()
    def CheckTime(self):
        return time.time() - self.startTime

    def PrintCheckTime(self, msg):
        elapsedTime = self.CheckTime()
        print msg + ' : ' + str(elapsedTime) + ' sec, ' + str(elapsedTime/60) + ' min'

class TrainingPlot():

    listXcoord = []
    listTrainLoss = []
    listTrainAcc = []
    listTestLoss = []
    listTestAcc = []
    iterSampleCount = 0
    iter = 0
    sumTestAcc = 0
    batchSize = 0
    sampleCount = 0
    totalIter = 0
    avgTestAcc = 0
    watchTotal = StopWatch()
    watchSingle = StopWatch()

    def __init__(self):

        plt.ion()
        plt.show()
        self.watchTotal.StartTime()
        self.watchSingle.StartTime()

    def SetConfig(self, batchSize, sampleCount, totalIter):
        self.batchSize = batchSize
        self.sampleCount = sampleCount
        self.totalIter = totalIter


    def AddTest(self, trainAcc, testAcc):
        #self.sampleCount += self.batchSize
        self.iter += 1
        self.listXcoord.append(self.iter)
        self.listTrainAcc.append(trainAcc)
        self.listTestAcc.append(testAcc)

        self.sumTestAcc += testAcc
        self.avgTestAcc = self.sumTestAcc/self.iter
        print ('#Iter %d / %d :TrainAcc %f, TestAcc %f, AvgAcc %f' %
               (self.iter, self.totalIter, trainAcc,testAcc,self.avgTestAcc))

    def Add(self, iter, trainLoss, testLoss, trainAcc, testAcc):
        self.sampleCount += self.batchSize
        self.iter = iter
        self.listXcoord.append(self.iter)
        self.listTrainLoss.append(trainLoss)
        self.listTrainAcc.append(trainAcc)
        self.listTestLoss.append(testLoss)
        self.listTestAcc.append(testAcc)
        totalElapsedTime = self.watchTotal.CheckTime()
        singleEapsedTime = self.watchSingle.CheckTime()
        self.watchSingle.StartTime()
        #print self.watch.startTime,elapsedTime
        totalEstimatedTime = (float(self.totalIter) / (float(self.iter) + 0.0001)) * totalElapsedTime
        print ('#Iter %d / %d : Train %f, Test %f, TrainAcc %f, TestAcc %f (1 iter %.02f sec, total %.02f min, %.02f hour remained)' %
               (self.iter, self.totalIter,trainLoss, testLoss, trainAcc,testAcc,
                singleEapsedTime, totalElapsedTime/60, (totalEstimatedTime - totalElapsedTime)/3600))


    def Show(self):
        plt.clf()
        plt.subplot(1,2,1)
        plt.title('Loss : %.03f' % (self.listTestLoss[-1]))
        if (len(self.listTrainLoss) > 0):
            plt.plot(self.listXcoord,self.listTrainLoss, color= '#%02x%02x%02x' % (58,146,204) )
        if (len(self.listTestLoss) > 0):
            plt.plot(self.listXcoord,self.listTestLoss, color= '#%02x%02x%02x' % (220,98,45))
        plt.legend(['train', 'val'])
        plt.subplot(1,2,2)
        plt.title('Acc : %.03f' % (self.listTestAcc[-1]))
        if (len(self.listTrainAcc) > 0):
            plt.plot(self.listXcoord,self.listTrainAcc, color= '#%02x%02x%02x' % (58,146,204) )
        if (len(self.listTestAcc) > 0):
            plt.plot(self.listXcoord,self.listTestAcc, color='#%02x%02x%02x' % (34,177,76))
        plt.ylim([0,1])

        plt.draw()
        plt.pause(0.001)
        plt.show()
