import numpy as np
import scipy.io as sio
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import glob
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import fill

count = [0,0,0,0,0,0,0,0,0,0,0,0]
stride = 10

class Param:
    resizeWidth = 128
    resizeHeight = 64
# roi : (left, top, right, bottom)
def SaveCropImage(imgData, classId, roi):
    savePath = 'data/Cropped/%d/%d.png' % (classId, count[classId])
    count[classId] = count[classId] + 1
    croppedImg = imgData.crop(roi)
    croppedImg = croppedImg.resize((32, 32), Image.BICUBIC)
    croppedImg.save(savePath)
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



def ShowImage(digitStruct, i):
    info = (digitStruct['digitStruct'][0])[i]
    filename = info[0][0]
    rois = info[1]
    # print filename
    imgData = Image.open(pathLoad + filename)
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
        left += width / 4
        right -= width / 4
        top += height / 4
        bottom -= height / 4
        classId = int(rois[0,j][4])
        gt = (left, top, right, bottom, width, height)
        SaveCropImage(imgData, classId, (left, top, right, bottom))
        gtList.append(gt)
        # print savePath, count

    NegativeSampleMining(imgData, gtList, stride)

    if (top - height > 0):
        SaveCropImage(imgData, 0, (left, top - height, right, bottom - height))

    if (top + height < imgData.size[1]):
        SaveCropImage(imgData, 0, (left, top + height, right, bottom + height))
        # print left, top, right, bottom
        # draw.rectangle([(left,top), (right,bottom)],fill=classId)
    if (0):
        # plt.ioff()
        plt.imshow(imgData)
        plt.imshow(imgLabel, vmin=0, vmax=10, alpha=0.4)
        plt.title(imgData.size)
        # plt.pause(0.1)
        plt.show()
    return imgData, imgLabel

def CreateDB(pathLoad, savePath):
    digitStruct = sio.loadmat(pathLoad + 'digitStruct.mat')
    m = digitStruct['digitStruct'].shape[1]
    print 'Load %d data' % m

    for i in range(m):

        progress = float(i) / float(m) * 100.0
        if i % 100 == 0:
            print '%d / %d : %.1f %%' % (i,m, progress)
        imgData, imgLabel = ShowImage(digitStruct, i)

        # newData = Image.new('RGB', (Param.resizeWidth, Param.resizeHeight))
        # newLabel = Image.new('L', (Param.resizeWidth, Param.resizeHeight))
        # width = imgData.size[0]
        # height = imgData.size[1]
        # aspectRatio = float(width) / float(height)
        # # smaller than resizWidth/Height, paste
        # if (width <= Param.resizeWidth and height <= Param.resizeHeight):
        #     resizeData = imgData
        #     resizeLabel = imgLabel
        # # keep aspect ratio
        # elif (width > height):
        #     resizeData = imgData.resize((Param.resizeWidth, (int)(Param.resizeHeight / aspectRatio)), Image.BICUBIC)
        #     resizeLabel = imgLabel.resize((Param.resizeWidth, (int)(Param.resizeHeight / aspectRatio)), Image.NEAREST)
        # else:
        #     resizeData = imgData.resize((int(Param.resizeWidth * aspectRatio), height), Image.BICUBIC)
        #     resizeLabel = imgLabel.resize((int(Param.resizeWidth * aspectRatio), height), Image.NEAREST)
        # # print resizeData.size
        # # print newData.size
        # newData.paste(resizeData, (0,0))
        # newData.save(savePath + '/%d.png' % i)
        # newLabel.paste(resizeLabel, (0,0))
        # newLabel.save(savePath + 'annot/%d.png' % i)


pathLoad = 'data/Original/train/'
savePath = 'data/PixelLabel/train'
CreateDB(pathLoad, savePath)


pathLoad = 'data/Original/val/'
savePath = 'data/PixelLabel/val'
CreateDB(pathLoad, savePath)

