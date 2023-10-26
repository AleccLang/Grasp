# Add in file path manually
# Press any key on keyboard to show next image step

from ctypes import sizeof
import imghdr
from itertools import accumulate
from locale import normalize
from operator import index
import cv2
import numpy as np
import math
import imutils

# Loading the image
image = cv2.imread('/home/alec/Images1/13917.png', 0)
image = image[16:-250, :]

scale_percent = 38
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
image= cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

height, width = image.shape
newWidth = width+200
newHeight = height+200
paddedImg = np.zeros((newHeight,newWidth))
xCenter = math.floor((newWidth - width)/2)  # Determine centre offsets. source: https://stackoverflow.com/questions/43391205/add-padding-to-images-to-get-them-into-the-same-shape.
yCenter = math.floor((newHeight - height)/2)
paddedImg[yCenter:yCenter+height, xCenter:xCenter+width] = image # Copy the image into the centre of the new padded image.
paddedImg = paddedImg.astype(np.uint8)
cv2.imshow("result", paddedImg)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("Original image", image)
cv2.waitKey(0)
cv2.destroyWindow("Histogram Equalisation")

imgNorm = cv2.normalize(paddedImg, None, alpha=800, beta=200, norm_type=cv2.NORM_MINMAX)

# Histogram equalise
img = cv2.equalizeHist(paddedImg)
img1 = cv2.equalizeHist(imgNorm)


cv2.imshow("Histogram Equalisation", img)
cv2.waitKey(0)
cv2.destroyWindow("Histogram Equalisation")

cv2.imshow("Histogram Normalisation", img1)
cv2.waitKey(0)
cv2.destroyWindow("Histogram Normalisation")

# Blur image to anti-alias the hand
# blur_img = cv2.GaussianBlur(img, (5, 5), 0)

# Threshold image
thresh = cv2.threshold(img, 170, 255, cv2.THRESH_BINARY)[1]

cv2.imshow("Thresholded image", thresh)
cv2.waitKey(0)
cv2.destroyWindow("Thresholded image")

# Apply the Component analysis function
analysis = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)
(numLabels, labels, stats, centroid) = analysis

# Extract largest component
componentSizes = stats[:, -1]
maxLabel = 1
maxSize = componentSizes[1]
for i in range(1, numLabels):
    if componentSizes[i] > maxSize:
        maxLabel = i
        maxSize = componentSizes[i]

handcomponent = np.zeros(paddedImg.shape)
handcomponent[labels == maxLabel] = 255

cv2.imshow("Biggest component", handcomponent)
cv2.waitKey(0)
cv2.destroyWindow("Biggest component")

# Need to change data type otherwise we get a format error when finding the contours
handcomponent = handcomponent.astype(np.uint8)


# Finds the hand contour
contours = cv2.findContours(handcomponent.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]


# Draws the hand-contour
handContourColour = cv2.cvtColor(handcomponent, cv2.COLOR_GRAY2BGR)
cv2.drawContours(handContourColour, contours, -1, (255, 0, 0), 4)

cv2.imshow("Hand Contour", handContourColour)
cv2.waitKey(0)
cv2.destroyWindow("Hand Contour")

# Finds the convex hull
convexHull = cv2.convexHull(contours[0])

# Draws the convex hull
cv2.drawContours(handContourColour, [convexHull], -1, (0, 255, 0), 4)

cv2.imshow('ConvexHull', handContourColour)
cv2.waitKey(0)
cv2.destroyWindow("Convex Hull")

hull = cv2.convexHull(contours[0], returnPoints= False) # Need to use indices of the points, so need to calculate hull again using 'returnPoints = False' as we can't use convexHull.
convexDefs = cv2.convexityDefects(contours[0], hull)
convexDefectList = [] # List of the convex defects.
for i in range(convexDefs.shape[0]):
    startIndex,endIndex,farIndex,depth = convexDefs[i,0]
    far = tuple(contours[0][farIndex][0])
    convexDefectList.append(far)
    cv2.circle(handContourColour,far,4,[0,0,255],-1) # Draws the convex defects as circles.


# Need to add something to negate when there are multiple convext defects on the tip of the middle finger.
convexDefectList.sort(key=lambda pair: pair[1]) # Sorting the list of defects to get the coordinates for the middle finger
midFingerCoord = convexDefectList[0] # The coordinates of middle finger.
convexDefectList.sort(key=lambda pair: pair[0]) # Sorts list based on x-coord.
indexMidFinger = convexDefectList.index(midFingerCoord)
for i in range(len(convexDefectList)):
    if(convexDefectList[indexMidFinger-i][1] > (convexDefectList[indexMidFinger][1])+50 and convexDefectList[indexMidFinger-i][1] < (convexDefectList[indexMidFinger][1])+250): # Error checking to make sure it doest pick up wrong convex points.
        indexLeftDefectPnt = indexMidFinger - i # Index of defect point to the left of the middle finger.
        break
    else:
        i = i - 1
for j in range(len(convexDefectList)):
    if(convexDefectList[indexMidFinger+j][1] > (convexDefectList[indexMidFinger][1])+50 and convexDefectList[indexMidFinger+j][1] < (convexDefectList[indexMidFinger][1])+250): # Error checking to make sure it doest pick up wrong convex points.
        indexRightDefectPnt = indexMidFinger + j # Index of defect point to the right of the middle finger.
        break
    else:
        j = j + 1       

print(convexDefectList[indexLeftDefectPnt])
print(convexDefectList[indexMidFinger])
print(convexDefectList[indexRightDefectPnt])

centroidX = 0 # X-coord of centroid of hand.
centroidY = 0 # Y-coord of centroid of hand.
M = cv2.moments(contours[0]) # Use Moments (The weighted average of pixel intensities) to find the centroid of the hand. (source: https://learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/)
centroidX = int(M['m10']/M['m00']) # X-coord of centroid of hand.
centroidY = int(M['m01']/M['m00']) # Y-coord of centroid of hand.
#cv2.circle(handContourColour, (centroidX, centroidY), 4, (0, 0, 255), -1) # Draws the centroid of the hand.

cv2.imshow("Covexity Defects",handContourColour)
cv2.waitKey(0)
cv2.destroyWindow("Covexity Defects")

# Centres the hand component before rotating it using the defect points calculated above.
height, width = paddedImg.shape[:2]
imgCentroidX = width/2 # X-coord centroid of image
imgCentroidY = height/2 # Y-coord centroid of image
shiftX = math.floor(imgCentroidX - centroidX) 
shiftY = math.floor(imgCentroidY - centroidY)
transMatrix = np.float32([[1, 0, -math.floor(centroidX-imgCentroidX)], [0, 1, -math.floor(centroidY-imgCentroidY)]])
img_translation = cv2.warpAffine(handContourColour, transMatrix, (width, height))

cv2.imshow('Translation', img_translation)
cv2.waitKey()
cv2.destroyWindow("Translation")


xLeft = convexDefectList[indexLeftDefectPnt][0]
yLeft = convexDefectList[indexLeftDefectPnt][1]
xRight = convexDefectList[indexRightDefectPnt][0]
yRight = convexDefectList[indexRightDefectPnt][1]
adj = xLeft - xRight
opp = yLeft - yRight
angle = math.degrees(math.tan(opp/adj)) # The angle made between the two defect points and the perpendicular.
print(angle)

a = np.array([convexDefectList[indexLeftDefectPnt][0],convexDefectList[indexLeftDefectPnt][1]])
b = np.array([convexDefectList[indexMidFinger][0],convexDefectList[indexMidFinger][1]])
c = np.array([convexDefectList[indexRightDefectPnt][0],convexDefectList[indexRightDefectPnt][1]])
ba = a - b
bc = c - b
cosine= np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
angle1 = np.degrees(np.arccos(cosine))/2 # The angle made at the middlefinger with the 2 defect points.
print(angle1)

if(angle < 0): # Determines which direction to rotate image based on the angle sign.
    rotateImg = imutils.rotate_bound(img_translation, angle1)
else:
    rotateImg = imutils.rotate_bound(img_translation, -angle1)
cv2.imshow('Rotation', rotateImg)
cv2.waitKey()
cv2.destroyWindow("Rotation")