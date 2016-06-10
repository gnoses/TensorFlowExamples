"""
Convolutional Encoder Decoder Net

Usage :
1. Download CamVid dataset ()
2. Run createDB once (Set following condition to 1)
# Create DB (run once)
if (0):

3. Reset condition to 0 and run training

"""

from SemanticSegmentationLib import *
import datetime
import os
dataPath = './CamVid/'

# mode = 0 : training, 1 : predict and visualize, 2 : predict and evaluate accuracy
# modelType = 0 : no pooling, 1 : max pool + mean unpool
modelName = {0:'NoPooling', 1:'MeanUnpooling'}
resultText = []
for modelType in range(0,2):
    savePath = modelName[modelType] + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    # savePath = './snapshot'
    os.mkdir(savePath)

    # training and evaluation
    Process(0, modelType, dataPath, savePath)
    text = Process(2, modelType, dataPath, savePath)
    # print savePath

    text = 'Model type : %d' % modelType + ' ' + text
    resultText.append(text)

    with open(savePath + '/info.txt', 'wt') as file:
        print >> file, 'Model : ' + modelName[modelType]
        print >> file, 'Result : ' + text

for text in resultText:
    print text
