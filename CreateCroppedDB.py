import numpy as np
import scipy.io as sio
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import glob
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import fill
import os
import shutil

count = [0,0,0,0,0,0,0,0,0,0,0,0]
stride = 5


def MakePath(pathData):
    try:
        shutil.rmtree(pathData)
    except:
        pass

    try:
        print 'create ' + pathData
        os.makedirs(pathData)
        # os.makedirs('/home/gnoses/Tensorflow/SVHN/data/Cropped/train')
        for i in range(0,11):
            os.makedirs(pathData + '/%d' % i)
    except:
        print 'create ' + pathData + ' failed'
        return False
    return True

# roi : (left, top, right, bottom)
def SaveCropImage(imgData, classId, roi):
    imgPath = savePath + '/%d/%d.png' % (classId, count[classId])
    count[classId] = count[classId] + 1
    croppedImg = imgData.crop(roi)
    croppedImg = croppedImg.resize((32, 32), Image.BICUBIC)
    print imgPath
    croppedImg.save(imgPath)
    # print savePath

# calulate IoU
def CalcIoU(a, b):

    width = min((a[2],b[2])) - max((a[0],b[0]))
    if width <= 0:
        return 0
    height  = min((a[3], b[3])) - max((a[1], b[1]))
    if height <= 0:
        return 0
    areaA = a[4] * a[5]
    areaB = b[4] * b[5]
    intersect = width * height
    return float(intersect) / float(areaA + areaB - intersect)

# crop negative samples with same size of gt in specific stride
def NegativeSampleMining(imgData, gtList, stride):
    width = gtList[0][4]
    height = gtList[0][5]

    for row in range(0,imgData.size[1]-height-1,stride):
        for col in range(0, imgData.size[0]-width-1, stride):
            hit = False
            imgDisp = imgData.copy()
            draw = ImageDraw.Draw(imgDisp)
            b = (col, row, col + width, row + height, width, height)
            for gt in gtList:
                a = (gt[0],gt[1],gt[0]+gt[2],gt[1]+gt[3],gt[2],gt[3])


                # print col, row, width, height, CalcIoU(a,b)
                draw.rectangle([(col, row), (col + width, row + height)])
                iou = CalcIoU(a, b)
                plt.title(iou)
                if (iou > 0.2):
                    hit = True
                    break

            if hit == False:
                draw.rectangle([(col,row),(col+width,row+height)])
                SaveCropImage(imgData, 0, (col, row, col + width, row + height))



            # plt.ioff()
            # plt.imshow(imgDisp)
            # plt.pause(0.05)



def LoadImage(digitStruct, i):
    info = (digitStruct['digitStruct'][0])[i]
    filename = info[0][0]
    rois = info[1]
    # print filename
    try:
        imgData = Image.open(pathLoad + filename)
    except:
        # print pathLoad + filename + ' failed'
        return
    # if filename == '10727.png':
    #     print '-------------------------------------------------------'
    # print pathLoad + filename + ' ok'
    imgLabel = Image.new('L', imgData.size, 0)
    gtList = []

    for j in range(rois.shape[1]):
        draw = ImageDraw.Draw(imgLabel)
        # height, left, top, width, label
        left = int(rois[0,j][1])
        top = int(rois[0,j][2])
        right = int(left) + int(rois[0,j][3])
        bottom = top + int(rois[0,j][0])
        width = (right - left - 1)
        height = (bottom - top - 1)

        classId = int(rois[0,j][4])
        gt = (left, top, right, bottom, width, height)
        SaveCropImage(imgData, classId, (left, top, right, bottom))
        gtList.append(gt)
        # print savePath, count

    NegativeSampleMining(imgData, gtList, stride)

    if (0):
        # plt.ioff()
        plt.imshow(imgData)
        plt.imshow(imgLabel, vmin=0, vmax=10, alpha=0.4)
        plt.title(imgData.size)
        # plt.pause(0.1)
        plt.show()
    return imgData, imgLabel


# load images with classId directory
def WriteData(pathLoad, saveFile):
    classList = glob.glob(pathLoad + '/*')
    # print classList
    trainfile = open(saveFile,'wt')

    for c in classList:
        classId = int(os.path.basename(c))
        # print classId
        imgList = glob.glob(c + '/*.png')
        for file in imgList:
            # img = Image.open(file)
            # print file, img.size
            trainfile.write(file + ' ' + str(classId) + '\n')

def CreateDB(pathLoad, savePath):
    digitStruct = sio.loadmat(pathLoad + 'digitStruct.mat')
    m = digitStruct['digitStruct'].shape[1]
    print 'Load %d data' % m

    for i in range(m):
        LoadImage(digitStruct, i)



savePath = 'data/Cropped/train'
if (MakePath(savePath) == False):
    exit(0)
pathLoad = 'data/Original/train/'
CreateDB(pathLoad, savePath)
WriteData(savePath, savePath + '.txt')

savePath = 'data/Cropped/val'
MakePath(savePath)
pathLoad = 'data/Original/val/'
CreateDB(pathLoad, savePath)
WriteData(savePath, savePath + '.txt')


