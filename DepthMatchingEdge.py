'''
Created on Oct 11, 2015

@author: Robert Washbourne
'''
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import sys

def CorrsWithEdge(slit0, slit1, windowSize, lagDis):
    '''input two 1D slices from stereo images and correlates 
    using block matching with blocks sized windowSize'''
    if (len(slit0.shape) != 1 or len(slit1.shape)!= 1):
        raise Exception("array1 and array2 are not 1D")
    
    if (len(slit0.shape) != len(slit1.shape)):
        raise Exception("array1 and array2 are not the same length")
    
    if (windowSize % 2 == 0):
        raise Exception("windowSize must be odd")
    
    length = len(slit0) #lag length
    
    sideBar = (windowSize - 1) / 2 #side bar to skip
    
    lag  = xrange(0-lagDis,lagDis+1)
    lagLength = len(lag)
    
    slitCorrelations = []
    firstLag = lag[0]
    
    for x in xrange(sideBar + (lagLength / 2), length - sideBar - (lagLength / 2)):
        box0 = slit0[(x - sideBar) : (x + sideBar)]
        
        correl = np.zeros(lagLength)
        for l in lag:
            box1 = slit1[(x - sideBar + l) : (x + sideBar + l)]
            
            sum11 = np.inner(box0,box0) #finding cross correlation
            sum22 = np.inner(box1,box1)
            sum12 = np.inner(box0,box1)
            
            corr = sum12 / (sum11*sum22)**0.5 #make correlation
            correl[l-firstLag] = corr
            
        slitCorrelations.extend([np.argmax(correl)-lagDis])
    return(slitCorrelations)    

#get the images
im0 = scipy.misc.imread("../d0.png", 1)
im1 = scipy.misc.imread("../d1.png", 1)
print("Read images...")
#resize the images
im0_small = scipy.misc.imresize(im0, 0.2, interp='bicubic', mode=None).astype(float)
im1_small = scipy.misc.imresize(im1, 0.2, interp='bicubic', mode=None).astype(float)
print("Resized images...")

#find the means of the images
mean0 = np.mean(im0_small)
mean1 = np.mean(im1_small)

#subtract the means from the small images
im0_small -= mean0
im1_small -= mean1
print("Computed means...")

imageLoopData = range(0,im0_small.shape[0])
progress = len(imageLoopData)-1
image = []
#repeat the program for each line of the images
for y in imageLoopData:
    #get a slit from both images
    im0 = im0_small[y,:] #all of x, 500 down y
    im1 = im1_small[y,:]     
    image.extend([CorrsWithEdge(im0, im1, 17, 20)])
    percent = round((float(y) / progress)*100, 2)
    sys.stdout.write("\r%d%%" % percent)
    sys.stdout.flush()

# threeDee = []    
# image = np.array(image) 
# for x in range(image.shape[0]):
#     for y in range(image.shape[1]):
#         threeDee.extend([[x,y,image[x,y]]])
# print(threeDee)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# 
# for point in threeDee:
#     print(point)
#     ax.scatter(point[0],point[1],point[2])
# 
# plt.show()
image = np.array(image)
distance = (im0_small.shape[1]-image.shape[1]) / 2
im0_small = im0_small[:,distance:im0_small.shape[1]-distance]
print(image[100][100])
im0_small[image>-5] = 0

print("\nDone, now plotting...")
print('Mean depth is '+ str(np.mean(image)))
maxes = np.amin(image)
print('Max depth is '+ str(maxes))
plt.subplot(1,3,1)
plt.imshow(image, cmap = "binary", interpolation='nearest')
plt.subplot(1,3,2)
plt.imshow(im0_small, cmap = "binary", interpolation='nearest')
plt.subplot(1,3,3)
plt.imshow(im1_small, cmap = "binary", interpolation='nearest')
plt.show()