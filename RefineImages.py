from PIL import Image
import glob
import matplotlib.pylab as plt
import os
import numpy as np

# showImage = True
showImage = False
classes = 10

def CreateDataList(categoryName):
    pathLoad1 = categoryName
    pathLoad2 = categoryName + 'annot'
    curPath = os.path.dirname(os.path.abspath(__file__))
    print curPath
    fileList1 = glob.glob(curPath + '/' + pathLoad1 + '/*.png')

    #fileList2 = glob.glob(pathLoad2 + '/*.png')
    #plt.ion()
    trainFile = open(categoryName + '.txt','wt')
    count = 0
    occupancyMap = {}
    for i in range(classes):
        occupancyMap[i] = 0

    for filename in fileList1:
        img1 = Image.open(filename)
        #filename2 = filename.replace('_IPMImg', '_IPMLabel')
        filename2 = curPath + '/' + pathLoad2 + '/' + os.path.basename(filename)
        img2 = Image.open(filename2)

        if (showImage):
            plt.subplot(1, 2, 1)
            plt.imshow(img1)
            plt.imshow(img2, alpha=0.4)
            plt.subplot(1, 2, 2)
            plt.imshow(img2)
            plt.show()

        pixel = np.array(img2)
        abnormal = False
        # if np.sum(pixel > 33):
        #     print (pixel[pixel > 33])
        for i in range(classes):
            occupancyMap[i] += np.sum(pixel == i)
            # if occupancy must be in [0.4, 0.6]
            occupancy = float(np.sum(pixel == i)) / (pixel.shape[0] * pixel.shape[1])
            #occupancyList.append(occupancy)

            #print filename, occupancy
            #break
            # if one class has too much occupancy
            # if (occupancy > 0.5):
            #     abnormal = True
            #     break
            #     print img2
            #     plt.subplot(1,2,1)
            #     plt.imshow(img1)
            #     plt.imshow(img2, alpha=0.4)
            #     plt.subplot(1,2,2)
            #     plt.imshow(img2)
            #     plt.title(occupancy)
            #     plt.draw()
            #     #plt.pause(1)
            #     plt.show()
        if abnormal == True:
            continue
        print >> trainFile, filename
        print >> trainFile, filename2

        #cropimg.save(pathSave + '/' + filename)
        #plt.title(pathCity)
        #plt.imshow(cropimg)
        #plt.show()
        count += 1
    print ('%d data list created' % count)
    trainFile.close()
    total = 0
    for i in range(classes):
        total += occupancyMap[i]
    for i in range(classes):
        print i, float(occupancyMap[i]) / total

    print(occupancyMap)
    return occupancyMap

CreateDataList('train')
CreateDataList('val')
#CreateDataList('test')
