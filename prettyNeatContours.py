import cv2
import numpy as np
import math
import imutils
import pandas as pd
images_data = {}
#import_list = ["13908.png","13909.png","13919.png","13902.png","13927.png","13898.png", "13930.png", "13902.png", "13955.png", "13993.png", "14145.png", "14037.png"]
import_list=["13908.png","13919.png"]
class Magic:
    modelHandCentroid=(0,0)

    def __init__(self, image, testCoords):
        self.image=image
        self.testCoords=testCoords
    def magic(self):

        image_data = {}

        original_image = cv2.imread(self.image, 0)
        original_image = original_image.astype(np.uint8)
        h, w = original_image.shape[:2]
        # Check to cut off bottom row
        for i in range(100):
            average_pixel_value = 0
            for j in range(w):
                average_pixel_value = average_pixel_value + original_image[h-1-i][w-1]
            average_pixel_value = average_pixel_value/w
            #print(average_pixel_value)
            if(average_pixel_value > 20):
                original_image = original_image[:-1, :]
            else:
                break
        
        height, width = original_image.shape[:2]
        uniform_width = 2000
        uniform_height = 2000
        padded_image = np.zeros((uniform_height,uniform_width))
        cx = math.floor((uniform_width - width)/2)  
        cy = math.floor((uniform_height - height)/2)
        padded_image[cy:cy+height, cx:cx+width] = original_image 
        padded_image = padded_image.astype(np.uint8)
        original_image = padded_image
        height, width = original_image.shape[:2]
        original_data = {"image":original_image,"additional":{}}
        image_data["original data"] = original_data

        cv2.imshow("Original Image", original_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        normalized_image = cv2.normalize(original_image, None, alpha=2200, beta=0, norm_type=cv2.NORM_MINMAX)
        normalized_data = {"image":normalized_image,"additional":{"alpha":2200,"beta":0,"type":"NORM_MINMAX"}}
        image_data["normalized data"] = normalized_data

        cv2.imshow("Normalized Image", normalized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        clahe = cv2.createCLAHE(3, (50,50))
        clahe_image = clahe.apply(original_image)
        clahe_data = {"image":clahe_image,"additional":{"clip":3,"grid":(50,50)}}
        image_data["clahe data"] = clahe_data

        cv2.imshow("Adaptive Histogram Equalized Image", clahe_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        average = math.floor(normalized_image.mean(axis=0).mean(axis=0))
        t1 = average+90
        binary_image = cv2.threshold(normalized_image, t1, 255, cv2.THRESH_BINARY)[1]
        binary_data = {"image":binary_image,"additional":{"average":average,"t1":t1,"t2":255}}
        image_data["binary data"] = binary_data

        cv2.imshow("Binary Image", binary_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        analysis = cv2.connectedComponentsWithStats(binary_image, 4, cv2.CV_32S)
        num_labels, labels, stats, centroid = analysis
        component_sizes = stats[:,-1]
        max_label = 1
        max_size = component_sizes[1]
        for i in range(1, num_labels):
            if component_sizes[i]>max_size:
                max_label = i
                max_size = component_sizes[i]
        hand_component = np.zeros(original_image.shape)
        hand_component[labels==max_label] = 255
        component_data = {"image":hand_component,"additional":{"label":labels[max_label],"centroid":centroid[max_label]}}
        image_data["component data"] = component_data

        cv2.imshow("Hand Component", hand_component)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        kernel1 = np.ones((10,10), np.uint8)
        kernel2 = np.ones((10,10), np.uint8)
        open_component = cv2.morphologyEx(hand_component, cv2.MORPH_OPEN, kernel1)
        close_component = cv2.morphologyEx(open_component, cv2.MORPH_CLOSE, kernel2)
        blur_component = cv2.GaussianBlur(close_component, (5,5), 0)
        smoothing_data = {"image":blur_component,"additional":{"open kernel":kernel1,"close kernel":kernel2,"blur kernel":(5,5),"open":open_component,"close":close_component}}
        image_data["smoothing data"] = smoothing_data

        cv2.imshow("Smoothed Hand Component", blur_component)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        blur_component = blur_component.astype(np.uint8)
        contours, hierarchy= cv2.findContours(blur_component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)                          
        if len(contours)>0:                
            areas = [cv2.contourArea(c) for c in contours]
            max_index = np.argmax(areas)                
            cnt = contours[max_index]
        
        #translating hand so contour centred around model hand


        
        epsilon = 0.0005*cv2.arcLength(cnt, True)            
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        mask = np.zeros(clahe_image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, contours, -1, 255, -1, cv2.LINE_AA)
        clahe_image = cv2.bitwise_and(clahe_image, clahe_image, mask=mask)
        contour_hand = clahe_image.copy()
        cv2.drawContours(contour_hand, contours, -1, (255,255,255), 4, cv2.LINE_AA)
        contour_data = {"image":contour_hand,"additional":{"contours":contours}}
        image_data["contour data"] = contour_data

        cv2.imshow("Contoured Hand", contour_hand)
        cv2.waitKey(0)
        cv2.destroyAllWindows()




        print("contour before")
        print(cnt[i])
        
    #Added identifying fingertips
    #------------------------------------------------------------------------------------------------------------

        hull = cv2.convexHull(approx, returnPoints=False)
        defects = cv2.convexityDefects(approx, hull)
        convex_image = contour_hand.copy()
        defect_coordinates = {}
        coord=[]
        distance=[]
        defect_coord = []
        finger_coord = []
        lengthXY=[]
        for i in range(defects.shape[0]):  
            #new function
            s, e, f, d = defects[i,0]
            far = tuple(approx[f][0])                 
            tempArr=[d,far[0],far[1]]
            lengthXY.append(tempArr)
            defect_data = {}
            s, e, f, d = defects[i,0]                
            start = tuple(approx[s][0])          
            end = tuple(approx[e][0])      
            far = tuple(approx[f][0]) 
            defect_data["start"] = start 
            defect_data["end"] = end 
            defect_data["far"] = far
            defect_data["distance"]=d 
            defect_data["y"]=int(far[1])
            defect_data["x"]=int(far[0])
            defect_coordinates[str(far)] = defect_data
            defect_coord.append(far)
            finger_coord.append(end)
            coord.append(far)
            distance.append(d)
        lengthXY.sort(key=lambda lengthXY:lengthXY[2],reverse=True)
        leftWrist=lengthXY.pop(0)
        rightWrist=lengthXY.pop(0)
        if(leftWrist[1]<rightWrist[1]):
            temp=leftWrist
            leftWrist=rightWrist
            rightWrist=temp 
        lengthXY.sort(key=lambda lengthXY:lengthXY[0],reverse=True)
        betweenFingerDefects=lengthXY[0:4]
        betweenFingerDefects.sort(key=lambda betweenFingerDefects:betweenFingerDefects[1],reverse=True)
        orderedFingerDefects=[]
        wristDefects=[leftWrist,rightWrist]
        orderedWristDefects=[]
        
        for i in range(len(wristDefects)):
            temp=wristDefects[i]
            tempTuple=(temp[1],temp[2])
            tempDist=temp[0]
            orderedWristDefects.append([tempDist,tempTuple])
        for i in range(len(betweenFingerDefects)):
            temp=betweenFingerDefects[i]
            tempTuple=(temp[1],temp[2])
            tempDist=temp[0]
            orderedFingerDefects.append([tempDist,tempTuple])

        thumbIndex=orderedFingerDefects.pop(0)
        indexMiddle=orderedFingerDefects.pop(0)
        middleRing=orderedFingerDefects.pop(0)
        ringPinky=orderedFingerDefects.pop(0)
        allDefectsOrdered=[orderedWristDefects[0],thumbIndex,indexMiddle,middleRing,ringPinky,orderedWristDefects[1]]
        


        tempCol=0
        for i in range(len(allDefectsOrdered)):
            
            #cv2.circle(convex_image,allDefectsOrdered[i][1], 10, [120,120,120], -1) 
            cv2.circle(convex_image,allDefectsOrdered[i][1], 10, [tempCol,tempCol,tempCol], -1) 
            tempCol=tempCol+30

        #------------------------------------------------------------------------------------------------------------------------------

        
        for i in range(defects.shape[0]):   
            defect_data = {}
            s, e, f, d = defects[i,0] 
            start = tuple(approx[s][0])          
            end = tuple(approx[e][0])      
            far = tuple(approx[f][0]) 
            defect_data["start"] = start 
            defect_data["end"] = end 
            defect_data["far"] = far 
            defect_coordinates[str(far)] = defect_data


        indexMiddle=allDefectsOrdered[0]
        middleRing=allDefectsOrdered[1]
        ringPinky=allDefectsOrdered[2]
        pinkyEdge=allDefectsOrdered[3]
        thumbIndex=allDefectsOrdered[5]
        #Calculate max distance between the ring and pinky
        #Order defects:
        
        accurateFingertipCoordinates=[]
        for i in range(len(allDefectsOrdered)-1):
        #for ht in range(1):
            maxDist=0
            
            furthestPointCoordinates=()   
            x1=allDefectsOrdered[i][1][0]
            x2=allDefectsOrdered[i][1][1]
            y1= allDefectsOrdered[i+1][1][0]
            y2=allDefectsOrdered[i+1][1][1]

            #Method for reformatting contour array
            contourRange=[]
            for j in range(cnt.shape[0]):            
                tupleCnt=(cnt[j][0][0],cnt[j][0][1])            
                if(tupleCnt==allDefectsOrdered[i][1]):
                    startContourIndex=j

                    
            for j in range(cnt.shape[0]):
                tupleCnt=(cnt[j][0][0],cnt[j][0][1])
                if(tupleCnt==allDefectsOrdered[i+1][1]):
                    endContourIndex=j
                    print("true",tupleCnt)
                    print(allDefectsOrdered[i+1])
            
            
                
            if(startContourIndex>endContourIndex):
                for y in range(startContourIndex,cnt.shape[0]):
                    contourRange.append(cnt[y])
                for k in range(0,endContourIndex+1):
                    contourRange.append(cnt[k])
            else:
                contourRange=cnt[startContourIndex:endContourIndex]
            
            for k in range(len(contourRange)):
                currentPoint=contourRange[k][0]
                z1=currentPoint[0]
                z2=currentPoint[1]
                indexPointDist= math.sqrt(((x1-z1)**2)+((x2-z2)**2))
                middlePointDist= math.sqrt(((y1-z1)**2)+((y2-z2)**2))
                tempDist= indexPointDist+middlePointDist
                if(tempDist>maxDist):
                    maxDist=tempDist
                    furthestPointCoordinates=currentPoint
            accurateFingertipCoordinates.append(tuple(furthestPointCoordinates))
            cv2.circle(convex_image, tuple(furthestPointCoordinates), 10, [255,255,255], -1)

        grad=255

    
        
        #Beginning of Sum squared errors
        #————————————————————————————————————————————————
        #Firstly, calc a guaranteed left wrist end point
        
        minDistance=9999999
        leftWristCalcCoords=()
        for i in range(cnt.shape[0]):
            if(cnt[i][0][0]<accurateFingertipCoordinates[2][0] and cnt[i][0][1]<allDefectsOrdered[0][1][1]):
                currentPoint=cnt[i][0]
                z1=currentPoint[0]
                z2=currentPoint[1]
                x1=allDefectsOrdered[0][1][0]
                x2=allDefectsOrdered[0][1][1]
                indexPointDist= math.sqrt(((x1-z1)**2)+((x2-z2)**2))
                if(minDistance>indexPointDist):
                    minDistance=indexPointDist
                    leftWristCalcCoords=(z1,z2)
        print(accurateFingertipCoordinates[2][0])
        print("left wrist coords")
        print(leftWristCalcCoords)
        cv2.circle(convex_image,leftWristCalcCoords, 30, [120,120,120], -1) 

        #Centroid
        a=allDefectsOrdered[0][1]
        #b=allDefectsOrdered[1][1]
        c=allDefectsOrdered[2][1]
        d=allDefectsOrdered[3][1]
        #e=allDefectsOrdered[4][1]
        f=leftWristCalcCoords

        xCentroid=int((a[0]+c[0]+d[0]+f[0])/4)
        yCentroid=int((a[1]+c[1]+d[1]+f[1])/4)
        handCentroid=(xCentroid, yCentroid)
        cv2.circle(convex_image,handCentroid, 5, [0,0,0], -1) 
        
        
        #Now reformat array to start at right wrist and end at left wrist
        sumErrorsRange=[]
        startErrorRange=-1
        endErrorRange=-1

        for j in range(cnt.shape[0]):            
            tupleCnt=(cnt[j][0][0],cnt[j][0][1])            
            if(tupleCnt==allDefectsOrdered[0][1]):
                startContourIndex=j
                break

                
        for j in range(cnt.shape[0]):
            tupleCnt=(cnt[j][0][0],cnt[j][0][1])
            if(tupleCnt==leftWristCalcCoords):
                endContourIndex=j
                break
            
        if(startContourIndex>endContourIndex):
            for y in range(startContourIndex,cnt.shape[0]):
                sumErrorsRange.append(cnt[y])
            for k in range(0,endContourIndex+1):
                sumErrorsRange.append(cnt[k])
        else:
            sumErrorsRange=cnt[startContourIndex:endContourIndex]
            
        #Now arrays to start at left wrist and end at right wrist

        defect_coord.sort(key=lambda pair: pair[1], reverse=True) # Sorting based on Y-value.
        if(defect_coord[0][0] < defect_coord[1][0]): # Identifies the left and right wrist defect.
            lwd = defect_coord[0]
            rwd = defect_coord[1]
        else:
            lwd = defect_coord[1]
            rwd = defect_coord[0]
        defect_coord.pop(0)
        defect_coord.pop(0)
        defect_coord.sort(key=lambda pair: pair[0]) # Sorts defects based on X-value.
        defect_coord.insert(0, lwd) # Inserts the left wrist defect to front of list.
        defect_coord.insert(len(defect_coord),rwd) # Inserts the right wrist defect to back of list.

        fingertips = []
        for i in range(len(finger_coord)): # Filters out the defects at the bottom of the hand.
            if(finger_coord[i][1] < (contour_hand.shape[0]*0.8)):
                fingertips.append(finger_coord[i]) # Adds fingertip points to new list.
        fingertips.sort(key=lambda pair: pair[0]) # Sorts defects based on X-value.
        #print(fingertips)
        #print(defect_coord)
        handSimilarityVect = [] # Stores the lengths between defect-points and fingertips.
        for i in range(5):
            length = math.sqrt((defect_coord[i][0]-fingertips[i][0])**2 + (defect_coord[i][1]-fingertips[i][1])**2)
            handSimilarityVect.append(length)
            length1 = math.sqrt((defect_coord[i+1][0]-fingertips[i][0])**2 + (defect_coord[i+1][1]-fingertips[i][1])**2)
            handSimilarityVect.append(length1)
        #print(handSimilarityVect)
    
        cv2.imshow("Convex Points", convex_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        moments = cv2.moments(cnt)
        cx = int(moments['m10']/moments['m00'])
        cy = int(moments['m01']/moments['m00'])
        features_image = convex_image.copy()
        cv2.circle(features_image, (cx,cy), 4, [255,255,255], -1)


        pair_distances = []
        dup_check = []
        for coordinate in defect_coordinates:
            far1 = defect_coordinates[coordinate]["far"]
            for coordinate in defect_coordinates:
                far2 = defect_coordinates[coordinate]["far"]
                if far1 != far2 and (far1,far2) not in dup_check:
                    dup_check.append((far1, far2))
                    dup_check.append((far2, far1))
                    distance = math.sqrt(((far2[0]-far1[0])**2)+((far2[1]-far1[1])**2))
                    pair_distances.append((far1, far2, distance))

        sorted_by_distance = sorted(pair_distances, reverse=False, key=lambda item: item[2])
        pair1 = sorted_by_distance[0]
        pair2 = sorted_by_distance[1]
        far1_1, far2_1, distance_1 = pair1
        far1_2, far2_2, distance_2 = pair2       
        dup_list = [far1_1, far2_1, far1_2, far2_2]
        nodup_list = list(dict.fromkeys(dup_list))
        sorted_by_x = sorted(nodup_list, key=lambda item: item[0])
        middle_left = sorted_by_x[1]
        middle_right = sorted_by_x[2]
        middle_mid_bottom = (int((middle_left[0]+middle_right[0])/2), int((middle_left[1]+middle_right[1])/2))
        middle_top_left = defect_coordinates[str(middle_left)]["start"]
        middle_top_right = defect_coordinates[str(middle_right)]["end"]
        middle_mid_top = (int((middle_top_left[0]+middle_top_right[0])/2), int((middle_top_left[1]+middle_top_right[1])/2))
        try:
            gradient = (middle_mid_top[1]-middle_mid_bottom[1])/(middle_mid_top[0]-middle_mid_bottom[0])
            c = -(gradient*middle_mid_bottom[0])+middle_mid_bottom[1]
            height, width = features_image.shape[:2]
            x = math.floor((height-c)/gradient)
        except ZeroDivisionError:
            height, width = features_image.shape[:2]
            x = middle_mid_top[0]
        cv2.line(features_image, middle_mid_top, (x,height), [255,255,255], 4)
        image_data["center coord"] = (cx,cy)

        cv2.imshow("Features Computed", features_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        height, width = features_image.shape[:2]
        centered_image = features_image.copy()
        imgCentroidX = width/2  
        imgCentroidY = height/2
        transMatrix = np.float32([[1, 0, -math.floor(cx-imgCentroidX)], [0, 1, -math.floor(cy-imgCentroidY)]])
        centered_image = cv2.warpAffine(features_image, transMatrix, (width, height))
        centered_data = {"image":centered_image,"additional":{"shift":(int(imgCentroidX),int(imgCentroidY))}}
        image_data["centered data"] = centered_data
        
        centered_component = cv2.warpAffine(blur_component, transMatrix, (width, height))
        centered_component_data = {"image":centered_component,"additional":{"shift":(int(imgCentroidX),int(imgCentroidY))}}
        image_data["centered component data"] = centered_component_data

        cv2.imshow("Centered Image", centered_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        cv2.imshow("Centered Component", centered_component)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("contour middle")
        print(cnt[i])
        rotated_image = centered_image.copy()
        rotated_component = centered_component.copy()

        a = np.array([middle_mid_top[0],middle_mid_top[1]]) # Mid finger point.
        b = np.array([(middle_left[0]+middle_right[0])//2,(middle_left[1]+middle_right[1])//2,]) # Mid-point between the two defect points.
        c = np.array([(middle_right[0]),(middle_left[1]+middle_right[1])//2]) # Point on the horizontal on the right.
        ba = a - b
        bc = c - b
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        rotation_angle = np.degrees(np.arccos(cosine))
        M = cv2.getRotationMatrix2D((imgCentroidX,imgCentroidY), (90-rotation_angle), 1.0)
        rotated_image = cv2.warpAffine(centered_image, M, (2000,2000))
        rotated_data = {"image":rotated_image,"additional":{"angle":-(90-rotation_angle)}}
        image_data["rotated data"] = rotated_data
        
        rotated_component = cv2.warpAffine(centered_component, M, (2000,2000))
        rotated_component_data = {"image":rotated_component,"additional":{"angle":-(90-rotation_angle)}}
        image_data["rotated component data"] = rotated_component_data
        
        
        #centered
        P = np.float32([[1, 0, self.testCoords[0]],[0, 1, self.testCoords[1]]])
        #P = np.float32([[1, 0,0],[0, 1, 0]])
        alexisCenteredComponent=rotated_component.copy()
        alexisFinalImage=cv2.warpAffine(alexisCenteredComponent,P, (2000,2000))
        contours, hierarchy= cv2.findContours(alexisFinalImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)                          
        if len(contours)>0:                
            areas = [cv2.contourArea(c) for c in contours]
            max_index = np.argmax(areas)                
            alexisContours = contours[max_index]


        images_data[self.image] = image_data
    
        
        colour = 30
        image_item = images_data[self.image]

        rotated_component = image_item["rotated component data"]["image"]

        rotated_component_colour = cv2.cvtColor(rotated_component, cv2.COLOR_GRAY2BGR)
        hsv = cv2.cvtColor(rotated_component_colour, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv,(0,0,1),(255,255,255)) # Mask to select only the white component.
        rotated_component_colour1 = rotated_component_colour.copy()
        rotated_component_colour1[mask>0] = (220,colour,80) # Changes the white to a colour.
        print("contour after")
        print(cnt[i])
        #cv2.imshow("Composition Image", composition_image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        alexisTestImage=np.zeros((2500,2500,3)).astype(np.uint8)
        for i in range(alexisContours.shape[0]):
            cv2.circle(alexisTestImage,alexisContours[i][0], 10, [255,255,255], -1)     
        
        for i in range(cnt.shape[0]):
            cv2.circle(alexisTestImage,cnt[i][0], 10, [120,120,120], -1) 
        cv2.imshow("Rotated Component", alexisTestImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 
        return [cnt, convex_image, rotated_image, sumErrorsRange, handCentroid, allDefectsOrdered, leftWristCalcCoords,alexisContours]


        for image in images_data:
            image_item = images_data[image]

            rotated_component = image_item["rotated component data"]["image"]

            rotated_component_colour = cv2.cvtColor(rotated_component, cv2.COLOR_GRAY2BGR)
            hsv = cv2.cvtColor(rotated_component_colour, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv,(0,0,1),(255,255,255)) # Mask to select only the white component.
            rotated_component_colour1 = rotated_component_colour.copy()
            rotated_component_colour1[mask>0] = (220,colour,80) # Changes the white to a colour.
            
            cv2.imshow("bleh", rotated_component_colour1)
            cv2.waitKey(0)
            cv2.destroyAllWindows() 
            
            
            colour = colour+60
        

#–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
#Start of the application
#Next step is to format the arrays so they are the same size
contoursImage=np.zeros((2500,2500,3)).astype(np.uint8)
#the second argument is irrelevant now
baseHand= Magic("13898.png",(0,0))

cntBase,b,baseRotatedimage,baseOrderedContour, baseCentroid, baseDefectPoints, bLeftWrist,g= baseHand.magic()
Magic.modelHandCentroid=baseCentroid
print(baseOrderedContour)
sizeBaseContour=len(baseOrderedContour)
for i in range(len(baseOrderedContour)):
    cv2.circle(contoursImage,baseOrderedContour[i][0], 1, [255,255,255], -1) 

largestSSE=0
entireSSEdataset=[]
for image in import_list:
    firstComparisonHand=Magic(image,(0,0))
    cntFirst, b, comparisonRotatedImage, firstComparisonOrderedContour, comparisonCentroid, firstComparisonDefectPoints, fcLeftWrist,g= firstComparisonHand.magic()
    print("centroid")
    print(baseCentroid, comparisonCentroid)
    print(firstComparisonDefectPoints)
    horizontalOffset=baseCentroid[0]-comparisonCentroid[0]
    verticalOffset=baseCentroid[1]-comparisonCentroid[1]
    print("hoz",horizontalOffset,verticalOffset)


    #cv2.warpAffine(comparisonRotatedImage, M, comparisonRotatedImage.shape[1],comparisonRotatedImage.shape[0])
    #for i in range(cntBase.shape[0]):
        #cv2.circle(contoursImage,cntBase[i][0], 10, [120,120,120], -1) 
    #for i in range(cntFirst.shape[0]):
        #cv2.circle(contoursImage,cntFirst[i][0], 10, [120,120,120], -1) 

    cv2.imshow("Rotated Component", contoursImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

    #FINAL START HERE:

    #translate all coordinates and contours
    for i in range(cntFirst.shape[0]):
        x=cntFirst[i][0][0]+horizontalOffset
        y=cntFirst[i][0][1]+verticalOffset
        cntFirst[i][0]=(x,y)

    for i in range(len(firstComparisonDefectPoints)):
        print("cool defect")
        #print(firstComparisonDefectPoints[i][0])
        newX=firstComparisonDefectPoints[i][1][0]+horizontalOffset
        newY=firstComparisonDefectPoints[i][1][1]+verticalOffset
        firstComparisonDefectPoints[i][1]=(newX,newY)
    comparisonCentroid= (comparisonCentroid[0]+horizontalOffset,comparisonCentroid[1]+verticalOffset)
    fcLeftWrist=(fcLeftWrist[0]+horizontalOffset,fcLeftWrist[1]+verticalOffset)


    #Now they're translated, so have to do the comparison

    cv2.circle(contoursImage,baseCentroid, 50, [120,120,120], -1) 
    cv2.circle(contoursImage,comparisonCentroid, 50, [120,120,120], -1) 

    #for i in range(len(firstComparisonOrderedContour)):
        #cv2.circle(contoursImage,firstComparisonOrderedContour[i][0], 10, [120,120,120], -1) 
    sumSquaredErrors=0

    #CONVERT CONTOUR TO RIGHT ORDER
    #First convert base contour
    #At some point this will be a function as it needs to be applied ro both contours
    #make this into a functiont that takes in start and end point and formats array

    newContour=[]
    startErrorRange=-1
    endErrorRange=-1
    startContourIndex=-1
    endContourIndex=-1

    #Reapply everything for first Comparison Ordered contour
    #Version using returns (i.e version assumes ordered, well formatted Contours have already been calculated)
    sizeFirstComparisonContour=len(newContour)
    arrayOfSquareErrors=[]
    pureSquaredErrorTerms=[]
    smallerContour=[]
    largerContour=[]
    deviationsFromMeanArray=[]
    deviationSquaredFromMeanArray=[]
    errors=[]
    sumErrors=0

    firstComparisonOrderedContourSize=len(firstComparisonOrderedContour)
    baseOrderedContourSize=len(baseOrderedContour)
    if(baseOrderedContourSize>=firstComparisonOrderedContourSize):
        smallerContour=firstComparisonOrderedContour
        largerContour=baseOrderedContour
    else:
        smallerContour=baseOrderedContour
        largerContour=firstComparisonOrderedContour

    ratio=len(largerContour)/len(smallerContour)
    for i in range(len(smallerContour)):
        correspondingPos=int(round(ratio*i))
        pointSmaller=(smallerContour[i][0][0],smallerContour[i][0][1])
        pointLarger=(largerContour[correspondingPos][0][0],largerContour[correspondingPos][0][1])
        tempDistance=math.dist(pointSmaller,pointLarger)
        sumErrors=sumErrors+tempDistance
        squareErrors=tempDistance**2
        if(squareErrors>largestSSE):
            largestSSE=squareErrors
        if(baseOrderedContourSize>=firstComparisonOrderedContourSize):
            tempArr=[pointSmaller,squareErrors]
            errorsArr=[pointSmaller,tempDistance]
        else:
            tempArr=[pointLarger,squareErrors]
            errorsArr=[pointLarger,tempDistance]
        
        pureSquaredErrorTerms.append(squareErrors)
        arrayOfSquareErrors.append(tempArr)
        errors.append(errorsArr)
        sumSquaredErrors=sumSquaredErrors+(squareErrors)
        #if(i%20==0):
            #cv2.line(contoursImage,pointSmaller, pointLarger, (0, 255, 0), thickness=2)
    mean=sumErrors/len(smallerContour)
    for i in range(len(arrayOfSquareErrors)):
        errors[i][1]=errors[i][1]-mean

    meanSquareErrors=sumSquaredErrors/len(smallerContour)
    pd.Series(pureSquaredErrorTerms).plot(kind="hist",bins=10)
    deviationSquaredFromMeanArray=arrayOfSquareErrors
    for i in range(len(deviationSquaredFromMeanArray)):
        deviationSquaredFromMeanArray[i][1]=deviationSquaredFromMeanArray[i][1]-mean

    arrayOfSquareErrors.sort(key=lambda arrayOfSquareErrors:arrayOfSquareErrors[1],reverse=True)





























    #------------------------------------------------------------------------------------------------------------------------------------------------
    # Colour Method 1
    #Ranks sse of points and allocates greatest the most red colour, smallest the most green colour
    
    half=len(arrayOfSquareErrors)/2
    red=(0,0,255)
    yellow=(0,255,255)
    green=(0,255,0)
    currentColour=red
    for i in range(math.floor(half)):
        incR=(yellow[0]-red[0])/half
        incG=(yellow[1]-red[1])/half
        incB=(yellow[2]-red[2])/half
        currentColour=(currentColour[0]+incR,currentColour[1]+incG,currentColour[2]+incB)
        #cv2.circle(contoursImage, arrayOfSquareErrors[i][0], 3, currentColour, -1)
    currentColour=yellow
    for i in range(math.floor(half),len(arrayOfSquareErrors)):
        incR=(green[0]-yellow[0])/half
        incG=(green[1]-yellow[1])/half
        incB=(green[2]-yellow[2])/half
        currentColour=(currentColour[0]+incR,currentColour[1]+incG,currentColour[2]+incB)
        #cv2.circle(contoursImage, arrayOfSquareErrors[i][0], 3, currentColour, -1)











    # Colour Method 2
    #Assigns colour based on formula 
    #array is ( squared errors  - mean )
    

    maxDisplacement=deviationSquaredFromMeanArray[0][1]
    minDisplacement=deviationSquaredFromMeanArray[0][1]
    print(deviationSquaredFromMeanArray)
    for i in range (1,len(deviationSquaredFromMeanArray)):
        if(deviationSquaredFromMeanArray[i][1]>maxDisplacement):
            maxDisplacement=deviationSquaredFromMeanArray[i][1]
        elif(deviationSquaredFromMeanArray[i][1]<minDisplacement):
            minDisplacement=deviationSquaredFromMeanArray[i][1]
    rangeDisplacement=maxDisplacement-minDisplacement
    for i in range(len(deviationSquaredFromMeanArray)):
        #Maps input array with specific range to output between 0 and 1
        funcColourMapping=(deviationSquaredFromMeanArray[i][1]-minDisplacement)/rangeDisplacement

        smallestDeviationColour=(255,255,255)
        largestDeviationColour= (0,0,255)
        tempColour=[0,0,0]
        for j in range(0,3):
            #larger=largestDeviationColour
            #smaller=smallestDeviationColour
            #if(larger<smaller):
                #temp
            
            tempColour[j]=funcColourMapping*(largestDeviationColour[j]-smallestDeviationColour[j])+smallestDeviationColour[j]
        currentColour=(tempColour[0],tempColour[1],tempColour[2])
        #cv2.circle(contoursImage, deviationSquaredFromMeanArray[i][0], 3, currentColour, -1)






#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

    # Colour Method 3
    #Assigns colour based on formula 
    #array is ( error - mean )
    
    half=len(arrayOfSquareErrors)/2
    red=(0,0,255)
    yellow=(0,255,255)
    green=(0,255,0)

    maxDisplacement=errors[0][1]
    minDisplacement=errors[0][1]
    
    for i in range (1,len(errors)):
        if(errors[i][1]>maxDisplacement):
            maxDisplacement=errors[i][1]
        elif(errors[i][1]<minDisplacement):
            minDisplacement=errors[i][1]
    rangeDisplacement=maxDisplacement-minDisplacement
    for i in range(len(errors)):
        #Maps input array with specific range to output between 0 and 1
        funcColourMapping=(errors[i][1]-minDisplacement)/rangeDisplacement

        smallestDeviationColour=(0,0,255)
        largestDeviationColour= (0,255,0)
        tempColour=[0,0,0]
        for j in range(0,3):
            #larger=largestDeviationColour
            #smaller=smallestDeviationColour
            #if(larger<smaller):
                #temp
            
            tempColour[j]=funcColourMapping*(largestDeviationColour[j]-smallestDeviationColour[j])+smallestDeviationColour[j]
        currentColour=(tempColour[0],tempColour[1],tempColour[2])
        #cv2.circle(contoursImage, arrayOfSquareErrors[i][0], 3, currentColour, -1)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#
    # Colour Method 3
    #Assigns colour based on formula 
    #(sum of squared errors) )
    
    half=len(arrayOfSquareErrors)/2
    red=(0,0,255)
    yellow=(0,255,255)
    green=(0,255,0)

    maxDisplacement=arrayOfSquareErrors[0][1]
    minDisplacement=arrayOfSquareErrors[0][1]
    
    for i in range (1,len(errors)):
        if(arrayOfSquareErrors[i][1]>maxDisplacement):
            maxDisplacement=arrayOfSquareErrors[i][1]
        elif(arrayOfSquareErrors[i][1]<minDisplacement):
            minDisplacement=arrayOfSquareErrors[i][1]
    rangeDisplacement=maxDisplacement-minDisplacement
    smallestDeviationColour=(0,255,0)
    largestDeviationColour= (0,0,255)
    for i in range(len(arrayOfSquareErrors)):
        #Maps input array with specific range to output between 0 and 1
        funcColourMapping=(arrayOfSquareErrors[i][1]-minDisplacement)/rangeDisplacement


        tempColour=[0,0,0]
        for j in range(0,3):
            #larger=largestDeviationColour
            #smaller=smallestDeviationColour
            #if(larger<smaller):
                #temp
            
            tempColour[j]=funcColourMapping*(largestDeviationColour[j]-smallestDeviationColour[j])+smallestDeviationColour[j]
        currentColour=(tempColour[0],tempColour[1],tempColour[2])
        #cv2.circle(contoursImage, arrayOfSquareErrors[i][0], 3, currentColour, -1)



cv2.imshow(" Square Errors (colour assigned in order) ", contoursImage)
cv2.waitKey(0)
cv2.destroyAllWindows() 


#Colouring method 2: Select static threhsold values and assign all points a colour based on what category they fall into 
