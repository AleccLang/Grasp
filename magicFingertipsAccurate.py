import cv2
import numpy as np
import math
import imutils

images_data = {}
import_list=["14145.png"]
#import_list = ["13908.png","13909.png","13919.png","13902.png","13927.png","13898.png", "13930.png", "13902.png", "13955.png", "13993.png", "14145.png", "14037.png"]
#import_list = ["14145.png"]
for import_image in import_list:
    image_data = {}

    original_image = cv2.imread(import_image, 0)
    original_image = original_image.astype(np.uint8)
    h, w = original_image.shape[:2]
    # Check to cut off bottom row
    for i in range(50):
        average_pixel_value = 0
        for j in range(w):
            average_pixel_value = average_pixel_value + original_image[h-1-i][w-1]
        average_pixel_value = average_pixel_value/w
        #print(average_pixel_value)
        if(average_pixel_value > 150):
            original_image = original_image[:-1, :]
        else:
            break
    #
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

        #cv2.circle(convex_image, far, 10, [255,255,255], -1) 
        #cv2.circle(convex_image, start,10 , [255,255,255], -1)
        #cv2.circle(convex_image, end, 5, [255,255,255], -1 )
    print("Defects summary")
    print("Defects number")
    print(len(lengthXY))
    print("All defects listed")
    print(lengthXY)
    
    lengthXY.sort(key=lambda lengthXY:lengthXY[2],reverse=True)
    leftWrist=lengthXY.pop(0)
    rightWrist=lengthXY.pop(0)
    
    if(leftWrist[1]<rightWrist[1]):
        temp=leftWrist
        leftWrist=rightWrist
        rightWrist=temp
    print("Left wrist")
    print(leftWrist)
    print("Right wrist")
    print(rightWrist)
    
    lengthXY.sort(key=lambda lengthXY:lengthXY[0],reverse=True)
    betweenFingerDefects=lengthXY[0:4]
    print("4 longest remaining defects")
    print(betweenFingerDefects)
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
    
    print("All defects ordered")
    print(allDefectsOrdered)
    tempCol=0
    for i in range(len(allDefectsOrdered)):
        
        #cv2.circle(convex_image,allDefectsOrdered[i][1], 10, [120,120,120], -1) 
        cv2.circle(convex_image,allDefectsOrdered[i][1], 10, [tempCol,tempCol,tempCol], -1) 
        tempCol=tempCol+30
    #cv2.circle(convex_image,allDefectsOrdered[1][1], 100, [0,0,0], -1) 
    #cv2.circle(convex_image,allDefectsOrdered[2][1], 100, [100,100,100], -1) 
    #-------------------------------------------------------------------------------------------

    #------------------------------------------------------------------------------------------------------------------------------
    print("Ordering defect points")

    print("coordinates")


    print("defect coordinates")
    print(far)
    print("contour first point")
    print(str(cnt[1][0]))

    
    for i in range(defects.shape[0]):   
        defect_data = {}
        s, e, f, d = defects[i,0] 
        print(s,e,f,d)       

        print("-________________________________________")
        print(approx[s][0])
        print(approx[e][0])
        print(approx[f][0])
        start = tuple(approx[s][0])          
        end = tuple(approx[e][0])      
        far = tuple(approx[f][0]) 
        print(start,end, far)
        defect_data["start"] = start 
        defect_data["end"] = end 
        defect_data["far"] = far 
        defect_coordinates[str(far)] = defect_data
        print("FAR_________________________")
        print(far)
        print("Start________________________")
        print(start)
        print("END_________")
        print(end)
        

    indexMiddle=allDefectsOrdered[0]
    middleRing=allDefectsOrdered[1]
    ringPinky=allDefectsOrdered[2]
    pinkyEdge=allDefectsOrdered[3]
    thumbIndex=allDefectsOrdered[5]
    #Calculate max distance between the ring and pinky
    
    print(cnt[1][0][0])

    minx=600000
    maxx=0
    miny=6000000
    maxy=0
    #Order defects:

    print("cala")
    print(cnt[0])
    print(cnt[1])
    print(cnt[2])
    print(cnt[3])

    for i in range(cnt.shape[0]):
        currentPoint=cnt[i][0]
        #print(currentPoint[0], currentPoint[1])
        if(int(currentPoint[0])>maxx):
            maxx=currentPoint[0]
        if(int(currentPoint[0])<minx):
            minx=currentPoint[0]
        if(int(currentPoint[1])>maxy):
            maxy=currentPoint[1]
        if(int(currentPoint[1])<miny):
            miny=currentPoint[1] 
        
      
        #print(currentPoint)
    accurateFingertipCoordinates=[]
    for i in range(len(allDefectsOrdered)-1):
    #for ht in range(1):
        maxDist=0
        
        furthestPointCoordinates=()   
        print("test inside")
        x1=allDefectsOrdered[i][1][0]
        x2=allDefectsOrdered[i][1][1]
        y1= allDefectsOrdered[i+1][1][0]
        y2=allDefectsOrdered[i+1][1][1]
        print(x1,x2)
        print(y1,y2)
        startContourIndex=50

        endContourIndex=80
        contourRange=[]
        for j in range(cnt.shape[0]):
            
            tupleCnt=(cnt[j][0][0],cnt[j][0][1])
            
            if(tupleCnt==allDefectsOrdered[i][1]):
                startContourIndex=j
                print("true",tupleCnt)
                print(allDefectsOrdered[i])
                
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
        print("abra")
        
        #for k in range(len(contourRange)):
            #print("hi")
            #print(contourRange[k])
        print(len(contourRange))
        for k in range(len(contourRange)):
            currentPoint=contourRange[k][0]
            #cv2.circle(convex_image, currentPoint, 50, [255,255,255], -1)
            offset=50
            #if(currentPoint[0]>(start[0]-offset) and currentPoint[0]>(start[0]+offset) and currentPoint[1]>(start[1]-offset) and currentPoint[1]<(start[1]+offset)):
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
        
        
    print(defects)
    print("furthest point")
    print(furthestPointCoordinates)
    print(x1)
    print(y1)
    print(x2)
    print(y2)
    print("xs")
    print(maxx)
    print(minx)
    print("ys")
    print(maxy)
    print(miny)
    print("Defects")

    grad=255
 
    #cv2.circle(convex_image, tuple(cnt[i][0]), 10, [grad,grad,grad], -1) 
    #grad=grad-0.02
#cv2.circle(convex_image, (148,1500), 50, [255,255,255], -1) 
#cv2.circle(convex_image, (1295,1670), 50, [255,255,255], -1)     


    #Centroid
    a=allDefectsOrdered[0][1]
    b=allDefectsOrdered[len(allDefectsOrdered)-1][1]
    c=accurateFingertipCoordinates[2]
    xCentroid=int((a[0]+b[0]+c[0])/3)
    yCentroid=int((a[1]+b[1]+c[1])/3)
    cv2.circle(convex_image,(xCentroid,yCentroid), 5, [0,0,0], -1) 

    print("Coordinates and distances ordered by y descending")
    #for i in range(len(coord)):
        #print("Distance",str(distance[i]),"Coord",coord[i])
    #print(defect_coord)  
    #print("defect data")
    print(defect_coordinates)  
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

    #cv2.imshow("Centered Image", centered_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    #cv2.imshow("Centered Component", centered_component)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

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

    cv2.imshow("Rotated Image", rotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    
    cv2.imshow("Rotated Component", rotated_component)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

    images_data[import_image] = image_data
  
    composition_image = np.zeros((2000,2000,3)).astype(np.uint8)
    colour = 30
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

        #contours, hierarchy= cv2.findContours(rotated_component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        #cv2.drawContours(composition_image, contours, -1, (255,colour,80), 4, cv2.LINE_AA)
        composition_image = cv2.addWeighted(composition_image,0.8,rotated_component_colour1,0.4,0)
        colour = colour+60

    cv2.imshow("Composition Image", composition_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
