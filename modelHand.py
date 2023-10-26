import cv2
import numpy as np
import math
import imutils
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)
class Process:

    def __init__(self, image):
        self.image=image
    def magic(self):
        original_image = cv2.imread(self.image, 0)
        original_image = original_image.astype(np.uint8)

        normalized_image = cv2.normalize(original_image, None, alpha=1800, beta=0, norm_type=cv2.NORM_MINMAX)


        clahe = cv2.createCLAHE(3, (50,50))
        clahe_image = clahe.apply(original_image)


        average = math.floor(normalized_image.mean(axis=0).mean(axis=0))+10 
        binary_image = cv2.threshold(normalized_image, average, 255, cv2.THRESH_BINARY)[1]


        # connected components 

        analysis = cv2.connectedComponentsWithStats(binary_image, 8, cv2.CV_32S)
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

        kernel1 = np.ones((10,10), np.uint8)
        kernel2 = np.ones((10,10), np.uint8)
        open_component = cv2.morphologyEx(hand_component, cv2.MORPH_OPEN, kernel1)
        close_component = cv2.morphologyEx(open_component, cv2.MORPH_CLOSE, kernel2)
        blur_component = cv2.GaussianBlur(close_component, (5,5), 0)


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


        hull = cv2.convexHull(approx, returnPoints=False)
        defects = cv2.convexityDefects(approx, hull)
        convex_image = contour_hand.copy()
        color = 30
        defect_coordinates = {}
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
            cv2.circle(convex_image, far, 10, [255,255,255], -1) 
            cv2.circle(convex_image, start, 6, [255,255,255], -1)
            cv2.circle(convex_image, end, 6, [255,255,255], -1 )
            color = color+20 


        moments = cv2.moments(cnt)
        cx = int(moments['m10']/moments['m00'])
        cy = int(moments['m01']/moments['m00'])
        centered_image = convex_image.copy()
        cv2.circle(centered_image, (cx,cy), 4, [255,255,255], -1)


        pair_distances = []
        dup_check = []
        for coordinate in defect_coordinates:
            start1 = defect_coordinates[coordinate]["start"]
            end1 = defect_coordinates[coordinate]["end"]
            far1 = defect_coordinates[coordinate]["far"]
            for coordinate in defect_coordinates:
                start2 = defect_coordinates[coordinate]["start"]
                end2 = defect_coordinates[coordinate]["end"]
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
        gradient = (middle_mid_top[1]-middle_mid_bottom[1])/(middle_mid_top[0]-middle_mid_bottom[0])
        c = -(gradient*middle_mid_bottom[0])+middle_mid_bottom[1]
        height, width = centered_image.shape[:2]
        x = math.floor((height-c)/gradient)
        cv2.line(centered_image, middle_mid_top, (x,height), [255,255,255], 4)


        height, width = centered_image.shape[:2]
        point_x = width/2
        point_y = height*0.65
        shift_x = math.floor(point_x-cx)
        shift_y = math.floor(point_y-cy)
        trans_matrix = np.float32([[1,0,-shift_x], [0,1,-shift_y]])
        centered_image = cv2.warpAffine(centered_image, trans_matrix, (width,height))


        rotated_image = centered_image.copy()

        a = np.array([middle_mid_top[0],middle_mid_top[1]]) # Mid finger point.
        b = np.array([(middle_left[0]+middle_right[0])//2,(middle_left[1]+middle_right[1])//2,]) # Mid-point between the two defect points.
        c = np.array([(middle_right[0]),(middle_left[1]+middle_right[1])//2]) # Point on the horizontal on the right.
        ba = a - b
        bc = c - b
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        rotation_angle = np.degrees(np.arccos(cosine))
        if(rotation_angle < 90):
            rotated_image = imutils.rotate_bound(centered_image, -(90-rotation_angle))
        else:
            rotated_image = imutils.rotate_bound(centered_image, -(90-rotation_angle))
        rotated_image =rotated_image.astype(np.uint8)
        orow, ocol= rotated_image.shape
        above=math.floor(((2500-orow)/2))
        below=math.ceil(((2500-orow)/2))
        left=math.floor(((2500-ocol)/2))
        right=math.ceil(((2500-ocol)/2))
        img1 = np.pad(rotated_image, ((above,below),(left,right)), mode='constant', constant_values=0)
        return img1
    


modelHand= Process("13898.png")
modelHand=modelHand.magic()


#modelHand=modelHand.astype(np.uint8)


overlay= Process("13909.png")
overlay=overlay.magic()
#overlay = overlay.astype(np.uint8)



cv2.imshow("Smoothed Hand Component", modelHand)
cv2.waitKey(0)
cv2.destroyAllWindows()


cv2.imshow("Smoothed Hand Component", overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(modelHand.shape)
print(overlay.shape)
#overlay=np.resize(overlay,(2085,1466))


added_image = cv2.addWeighted(modelHand,0.8,overlay,0.4,0)

#arr1=np.zeros(2100,1500)
#for i in arr1: 
#added_image = cv2.addWeighted(overlay,0.4,overlay,0.1,0)
cv2.imshow("Superimposed hands", added_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
