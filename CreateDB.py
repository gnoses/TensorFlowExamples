import numpy as np
import scipy.io as sio
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import glob
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import fill

class Param:
    resizeWidth = 128
    resizeHeight = 128

def ShowImage(digitStruct, i):
    info = (digitStruct['digitStruct'][0])[i]
    filename = info[0][0]
    rois = info[1]
    # print filename
    imgData = Image.open(pathLoad + filename)
    imgLabel = Image.new('L', imgData.size, 0)
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
        # print left, top, right, bottom
        draw.rectangle([(left,top), (right,bottom)],fill=classId)
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
        imgData, imgLabel = ShowImage(digitStruct, i)
        newData = Image.new('RGB', (Param.resizeWidth, Param.resizeHeight))
        newLabel = Image.new('L', (Param.resizeWidth, Param.resizeHeight))
        width = imgData.size[0]
        height = imgData.size[1]
        aspectRatio = float(width) / float(height)
        # smaller than resizWidth/Height, paste
        if (width <= Param.resizeWidth and height <= Param.resizeHeight):
            resizeData = imgData
            resizeLabel = imgLabel
        # keep aspect ratio
        elif (width > height):
            resizeData = imgData.resize((Param.resizeWidth, (int)(Param.resizeHeight / aspectRatio)), Image.BICUBIC)
            resizeLabel = imgLabel.resize((Param.resizeWidth, (int)(Param.resizeHeight / aspectRatio)), Image.NEAREST)
        else:
            resizeData = imgData.resize((int(Param.resizeWidth * aspectRatio), height), Image.BICUBIC)
            resizeLabel = imgLabel.resize((int(Param.resizeWidth * aspectRatio), height), Image.NEAREST)
        # print resizeData.size
        # print newData.size
        newData.paste(resizeData, (0,0))
        newData.save(savePath + '/%d.png' % i)
        newLabel.paste(resizeLabel, (0,0))
        newLabel.save(savePath + 'annot/%d.png' % i)


pathLoad = 'data/Original/train/'
savePath = 'data/PixelLabel/train'
CreateDB(pathLoad, savePath)


pathLoad = 'data/Original/test/'
savePath = 'data/PixelLabel/val'
CreateDB(pathLoad, savePath)

