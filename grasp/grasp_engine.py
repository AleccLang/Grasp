import cv2
import numpy as np
from PyQt5.QtGui import QPixmap, QImage
import math
from matplotlib import pyplot as plt
import os


class GraspEngine:

    """
    GraspEngine class contains all the methods necessary for the proccessing and evaluation of images.
    
    ...
    
    Methods
    -------
    convertToArray(image_path)
        Reads in and converts image to numpy array.
    
    removeStupidLine(image_data, settings)
        Removes line of pixels at bottom of problem images.

    convertToUniformSize(image_data)
        Pads image with pixels so that all images are of equal size.

    convert_cv_qt(cv_img)
        Converts image in opencv format to image in pyqt5 format.
    
    maskedCLAHE(image_data, contours, settings)
        Creates a CLAHE (Contrast Limited Adaptive Histogram Equalization) of the original image.
    
    justNormalization(image_data, settings)
        Normalizes the input image.
            
    imageThreshold(image_data, settings)
        Produces a binary image based on 2 input thresholds.
        
    connectedComponents(image_data, settings)
        Determines and highlights the connected componets in the binary image.
    
    smoothEdges(hand_component, settings)
        Smooths the outer edges of the hand component.

    extractContours(image_data)
        Extracts the contour from the hand component.

    drawContourOverImage(image_data, contours)
        Draws the hand contour onto the input image.
    
    convexityDefects(image_data, contours)
        Calculates the convexity defects of the hand.
    
    identifyDefects(defect_coordinates)
        Identifies the coordinates of the finger-join and wrist defects.
    
    identifyPalmCentroid(image_data, finger_joins, wrist_defects)
        Finds and draws the centroid of the hands palm.
    
    arrangeContourPoints(contours, wrist_defects)
        Orders the contour points of the hand.
    
    identifyFingerTips(image_data, finger_joins, wrist_defects, contours)
        Calculates the coordinates of the finger tips and draws them.
    
    identifyAlignmentVector(image_data, finger_tips, finger_joins)
        Calculates and draws the alignment vector that will be used when rotating the hand.

    centerHand(image_data, centroid)
        Centres the hand to with the centre of the image.
    
    showAlignmentAccuracy(components)
        Layers the centred hands ontop of eachother to view the accuracy of alignment.

    rotateHand(image_data, finger_joins, finger_tips)
        Rotates the centred hand so that the middle finger points up.
    
    spikyHand(image_data, finger_joins, finger_tips, wrist_defects)
        Calculated the lenths between the fingertips, finger-joins and wrist defects and packs them into a vector.

    generateContourComparisonData(base_contour, comparison_contour)
        xxxxxx.

    drawBySquareError(image_data, array_of_square_errors)
        xxxxxx.
    
    drawBySquaredErrorMeanDiff(image_data, deviation_squared_from_mean_array)
        xxxxxx.
    
    drawByErrorMeanDiff(image_data, array_of_square_errors, errors)
        xxxxxx.

    drawBySumSquaredErrors(image_data, array_of_square_errors, errors)
        xxxxxx.

    kMeans(data)
        Creates a k-means cluster from 2D hand data points (boundry lengths and area) as a percentage of the ideal hand.
    
    handComp(hands, ideal_hand):
        Compares the area and hand boundry lengths of each hand to the ideal hand.
"""

    def convertToArray(image_path):
        """Reads in and converts png image to a numpy array.
        
        Parameters
        ----------
        cv_img : Contains the png image.
        
        Returns
        -------
        image : The converted image as a numpy array."""

        image = cv2.imread(image_path, 0)
        image = cv2.imread(image_path, 0)
        image = image.astype(np.uint8)
        return image

    def removeStupidLine(image_data, settings):
        """Removes line of pixels at bottom of problem images.
        
        Parameters
        ----------
        image_data : Contains the image.
        
        Returns
        -------
        image_data : The cropped image.
        rows_removed: Number of rows removed as a string"""

        if settings == ():
            (h, w) = image_data.shape[:2]
            rows_removed = 0
            for i in range(200):
                average_pixel_value = 0
                for j in range(w):
                    average_pixel_value = average_pixel_value \
                        + image_data[h - 1 - i][w - 1]
                average_pixel_value = average_pixel_value / w
                if average_pixel_value > 15:
                    image_data = image_data[:-1, :]
                    rows_removed += 1
                else:
                    break
            for i in range(20):
                average_pixel_value = 0
                for j in range(w):
                    average_pixel_value = average_pixel_value \
                        + image_data[i][w - 1]
                average_pixel_value = average_pixel_value / w
                if average_pixel_value > 15:
                    image_data = image_data[1:, :]
                    rows_removed += 1
                else:
                    break
        else:
            (row_count, ) = settings
            row_count = int(row_count)
            image_data = image_data[:-row_count, :]
            rows_removed = row_count
        return (image_data, (str(rows_removed), ))

    def convertToUniformSize(image_data):
        """Pads image with pixels so that all images are of equal size.
        
        Parameters
        ----------
        image_data : Contains the image.
        
        Returns
        -------
        padded_image : The uniformly sized image."""

        (height, width) = image_data.shape[:2]
        uniform_width = 2500
        uniform_height = 2500
        padded_image = np.zeros((uniform_height, uniform_width))
        cx = math.floor((uniform_width - width) / 2)
        cy = math.floor((uniform_height - height) / 2)
        padded_image[cy:cy + height, cx:cx + width] = image_data
        padded_image = padded_image.astype(np.uint8)
        return padded_image

    def convert_cv_qt(cv_img):
        """Converts image in opencv format to image in pyqt5 format.
        
        Parameters
        ----------
        cv_img : Contains the image in opencv format.
        
        Returns
        -------
        image : The converted image in pyqt5 format."""

        rgb_image = cv2.cvtColor(cv_img.astype(np.uint8),
                                 cv2.COLOR_BGR2RGB)
        (h, w, ch) = rgb_image.shape
        bytes_per_line = ch * w
        image = QImage(rgb_image.data, w, h, bytes_per_line,
                       QImage.Format_RGB888)
        return QPixmap.fromImage(image)

    def maskedCLAHE(image_data, contours, settings):
        """Creates a CLAHE (Contrast Limited Adaptive Histogram Equalization) of the original image.
        
        Parameters
        ----------
        image_data : Contains the image in opencv format.
        contours : Contour of the hand.
        settings : Custom user settings
        
        Returns
        -------
        clahe_image : The CLAHE image."""

        clip = 3
        x = 50
        y = 50
        if settings != ():
            (clip, x, y) = settings
            clip = int(clip)
            x = int(x)
            y = int(y)
        clahe = cv2.createCLAHE(clip, (x, y))
        clahe_image = clahe.apply(image_data)
        mask = np.zeros(clahe_image.shape[:2], dtype=np.uint8)
        cv2.drawContours(
            mask,
            contours,
            -1,
            255,
            -1,
            cv2.LINE_AA,
            )
        clahe_image = cv2.bitwise_and(clahe_image, clahe_image,
                mask=mask)
        return (clahe_image, (str(clip), str(x), str(y)))

    def justNormalization(image_data, settings):
        """Normalizes the input image.
        
        Parameters
        ----------
        image_data : Contains the image as a numpy array.
        settings : Custom user settings.
        
        Returns
        -------
        normalized_image : The normalized image."""

        alpha = 2200
        beta = 0
        if settings != ():
            (alpha, beta) = settings
            alpha = int(alpha)
            beta = int(beta)
        normalized_image = cv2.normalize(image_data, None, alpha=alpha,
                beta=beta, norm_type=cv2.NORM_MINMAX)
        return (normalized_image, (str(alpha), str(beta)))

    def imageThreshold(image_data, settings):
        """Produces a binary image based on 2 input thresholds.
        
        Parameters
        ----------
        image_data (ndarry): Multi-dimensional array containing the normalised image.
        
        settings : 2 user inputs for custom thresholds.
        
        Returns
        -------
        result_image_data (ndarry): Multi-dimensional array containing the binary thresholded image."""

        thresh = 180
        if settings != ():
            (thresh, ) = settings
            thresh = int(thresh)
        image = cv2.cvtColor(image_data, cv2.COLOR_GRAY2RGB)
        binary_image = cv2.threshold(image_data, thresh, 255,
                cv2.THRESH_BINARY)[1]
        return (binary_image, (str(thresh), ))

    def connectedComponents(image_data):
        """Determines and highlights the connected componets in the binary image.
        
        Parameters
        ----------
        image_data (ndarry): Multi-dimensional array containing the binary thresholded image.
        
        Returns
        -------
        result_image_data (ndarry): Multi-dimensional array containing the histogram normalised image.
        
        analysis : Component data, labels and centroids."""

        result_image_data = image_data.copy()
        analysis = cv2.connectedComponentsWithStats(image_data, 4,
                cv2.CV_32S)
        (num_labels, labels, stats, centroids) = analysis
        component_sizes = stats[:, -1]
        max_label = 1
        max_size = component_sizes[1]
        for i in range(1, num_labels):
            if component_sizes[i] > max_size:
                max_label = i
                max_size = component_sizes[i]
        hand_component = np.zeros(image_data.shape)
        hand_component[labels == max_label] = 255
        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            (cX, cY) = centroids[i]
            if i == max_label:  # draws a rectangle around the largest component.
                cv2.rectangle(result_image_data, (x, y), (x + w, y
                              + h), (173, 173, 173), 6)
            else:
                cv2.rectangle(result_image_data, (x, y), (x + w, y
                              + h), (81, 81, 81), 2)

        return (result_image_data, analysis, area, hand_component, (cX,
                cY))

    def smoothEdges(hand_component, settings):
        """Smooths the outer edges of the hand component.
        
        Parameters
        ----------
        hand_component : Contains binary hand component.
        settings : Custom user settings.
        
        Returns
        -------
        blur_component : The smoothed binary hand component."""

        ox = 10
        oy = 10
        cx = 10
        cy = 10
        gx = 5
        gy = 5
        if settings != ():

            (
                ox,
                oy,
                cx,
                cy,
                gx,
                gy,
                ) = settings
            ox = int(ox)
            oy = int(oy)
            cx = int(cx)
            cy = int(cy)
            gx = int(gx)
            gy = int(gy)
        kernel1 = np.ones((ox, oy), np.uint8)
        kernel2 = np.ones((cx, cy), np.uint8)
        open_component = cv2.morphologyEx(hand_component,
                cv2.MORPH_OPEN, kernel1)
        close_component = cv2.morphologyEx(open_component,
                cv2.MORPH_CLOSE, kernel2)
        blur_component = cv2.GaussianBlur(close_component, (gx, gy), 0)
        return (blur_component, (
            str(ox),
            str(oy),
            str(cx),
            str(cy),
            str(gx),
            str(gy),
            ))

    def extractContours(image_data):
        """Extracts the contour from the hand component.
        
        Parameters
        ----------
        image_data : Multi-dimensional array containing the largest component.
        
        Returns
        -------
        contours : The contour of the hand."""

        image_data = image_data.astype(np.uint8)
        (contours, hierarchy) = cv2.findContours(image_data,
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return contours

    def drawContourOverImage(image_data, contours):
        """Draws the hand contour onto the input image.
        
        Parameters
        ----------
        image_data : Contains the image to draw the contour on (multiple used; CLAHE, binary component, etc).
        contours : The contour of the hand
        
        Returns
        -------
        image_data : The converted image with the hand contour drawn onto it."""

        cv2.drawContours(
            image_data,
            contours,
            -1,
            (255, 255, 255),
            4,
            cv2.LINE_AA,
            )
        return image_data

    def convexityDefects(image_data, contours):
        """Calculates the convexity defects of the hand.
        
        Parameters
        ----------
        image_data : Contains binary hand component.
        contours : The contour of the hand.
        
        Returns
        -------
        convex_image : The binary hand with convexity defects drawn on.
        defect_coordinates : Coordinates of the defects.
        defects : Defects of the binary hand component"""

        convex_image = image_data.copy()
        epsilon = 0.0005 * cv2.arcLength(contours[-1], True)
        approx = cv2.approxPolyDP(contours[-1], epsilon, True)
        convex_hull = cv2.convexHull(approx, returnPoints=False)
        defects = cv2.convexityDefects(approx, convex_hull)
        defect_coordinates = []
        for i in range(defects.shape[0]):
            (s, e, f, d) = defects[i, 0]
            start = tuple(approx[s][0])
            end = tuple(approx[e][0])
            far = tuple(approx[f][0])
            defect_coordinates.append((start, far, end))
            cv2.circle(convex_image, far, 10, [255, 255, 255], -1)
            cv2.circle(convex_image, start, 5, [255, 255, 255], -1)
            cv2.circle(convex_image, end, 5, [255, 255, 255], -1)
        return (convex_image, defect_coordinates, defects)

    def identifyDefects(defect_coordinates):
        """Identifies the coordinates of the finger-join and wrist defects.
        
        Parameters
        ----------
        defect_coordinates : Contains the defect coordinates of the hand.
        
        Returns
        -------
        finger_joins : The coordinates of the finger-join defects.
        wrist_defects : The coordinates of the wrist defects."""

        finger_joins = {}
        finger_defects = []
        others = []
        for coordinate in defect_coordinates:
            (start, far, end) = coordinate
            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1]
                          - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1]
                          - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1])
                          ** 2)
            try:
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b
                                  * c)) * 57.2858
                if angle < 90:
                    finger_defects.append(far)
                else:
                    others.append(far)
            except ValueError:
                others.append(far)
        sorted_by_x = sorted(finger_defects, key=lambda item: item[0])
        finger_joins['pinky-ring'] = sorted_by_x[0]
        finger_joins['ring-middle'] = sorted_by_x[1]
        finger_joins['middle-index'] = sorted_by_x[2]
        finger_joins['index-thumb'] = sorted_by_x[3]
        if len(others) > 2:
            sorted_by_y = sorted(others, key=lambda item: item[1],
                                 reverse=True)
            others = [sorted_by_y[0], sorted_by_y[1]]
        sorted_by_x = sorted(others, key=lambda item: item[0])
        wrist_left = sorted_by_x[0]
        wrist_right = sorted_by_x[1]
        wrist_defects = {'wrist-left': wrist_left,
                         'wrist-right': wrist_right}
        return (finger_joins, wrist_defects)

    def identifyAccurateWristPoint(
        image_data,
        contours,
        wrist_defects,
        finger_tips,
        ):

        middle = finger_tips['middle']
        wrist_right = wrist_defects['wrist-right']
        min_dist = 9999999
        accurate_coords = ()
        for i in range(contours[-1].shape[0]):
            if contours[-1][i][0][0] < middle[0] \
                and contours[-1][i][0][1] < wrist_right[1]:
                current_point = contours[-1][i][0]
                z1 = current_point[0]
                z2 = current_point[1]
                x1 = wrist_right[0]
                x2 = wrist_right[1]
                dist = math.sqrt((x1 - z1) ** 2 + (x2 - z2) ** 2)
                if min_dist > dist:
                    min_dist = dist
                    accurate_coords = (z1, z2)
        image_data = cv2.circle(image_data, accurate_coords, 20, [255,
                                255, 255], -1)
        return (image_data, accurate_coords)

    def identifyPalmCentroid(image_data, finger_joins, wrist_defects):
        """Finds and draws the centroid of the hands palm.
        
        Parameters
        ----------
        image_data : Contains the binary hand.
        finger_joins : The coordinates of the finger-join defects.
        wrist_defects : The coordinates of the wrist defects.
        
        Returns
        -------
        image_data : The hand with the centroid drawn on.
        hand_centroid : The centroid of the hand."""

        wrist_left = wrist_defects['wrist-left']
        wrist_right = wrist_defects['wrist-right']
        ring_middle = finger_joins['ring-middle']
        middle_index = finger_joins['middle-index']
        hand_centroid = (int((wrist_left[0] + wrist_right[0]
                         + ring_middle[0] + middle_index[0]) / 4),
                         int((wrist_left[1] + wrist_right[1]
                         + ring_middle[1] + middle_index[1]) / 4))
        cv2.circle(image_data, hand_centroid, 15, [255, 255, 255], -1)
        return (image_data, hand_centroid)

    def arrangeContourPoints(contours, wrist_defects):
        """Orders the contour points of the hand.
        
        Parameters
        ----------
        contours : Contains the contour of the hand.
        wrist_defects : The coordinates of the wrist defects.
        
        Returns
        -------
        contour_range : the list of ordered contour points."""

        start_index = 0
        end_index = 0
        wrist_right = wrist_defects['wrist-right']
        wrist_left = wrist_defects['wrist-left']
        for j in range(contours[-1].shape[0]):
            point = (contours[-1][j][0][0], contours[-1][j][0][1])
            if point == wrist_right:
                start_index = j
            elif point == wrist_left:
                end_index = j
        contour_range = []
        if start_index > end_index:
            for i in range(start_index, contours[-1].shape[0]):
                contour_range.append(contours[-1][i])
            for k in range(0, end_index + 1):
                contour_range.append(contours[-1][k])
        else:
            contour_range = (contours[-1])[start_index:end_index]
        return contour_range

    def identifyFingerTips(
        image_data,
        finger_joins,
        wrist_defects,
        contours,
        ):
        """Calculates the coordinates of the finger tips and draws them.
        
        Parameters
        ----------
        image_data : Contains the hand.
        finger_joins : The coordinates of the finger-join defects.
        wrist_defects : The coordinates of the wrist defects. 
        contours : The contour of the hand.
        
        Returns
        -------
        image_data : The hand with the finger tip coordinates drawn on.
        finger_tips : The coordinates of the finger tips."""

        wrist_left = wrist_defects['wrist-left']
        wrist_right = wrist_defects['wrist-right']
        ring_middle = finger_joins['ring-middle']
        middle_index = finger_joins['middle-index']
        pinky_ring = finger_joins['pinky-ring']
        index_thumb = finger_joins['index-thumb']
        finger_defects = [('pinky', wrist_left, pinky_ring), ('ring',
                          pinky_ring, ring_middle), ('middle',
                          ring_middle, middle_index), ('index',
                          middle_index, index_thumb), ('thumb',
                          index_thumb, wrist_right)]
        finger_tips = {}
        for defect in finger_defects:
            (finger, left, right) = defect
            right_index = 0
            left_index = 0
            for j in range(contours[-1].shape[0]):
                point = (contours[-1][j][0][0], contours[-1][j][0][1])
                if point == right:
                    right_index = j
                elif point == left:
                    left_index = j
            contour_range = []
            if right_index > left_index:
                for i in range(right_index, contours[-1].shape[0]):
                    contour_range.append(contours[-1][i])
                for k in range(0, left_index + 1):
                    contour_range.append(contours[-1][k])
            else:
                contour_range = (contours[-1])[right_index:left_index]
            max_dist = 0
            furthest_point = 0
            for k in range(len(contour_range)):
                current_point = contour_range[k][0]
                left_dist = math.sqrt((left[0] - current_point[0]) ** 2
                        + (left[1] - current_point[1]) ** 2)
                right_dist = math.sqrt((right[0] - current_point[0])
                        ** 2 + (right[1] - current_point[1]) ** 2)
                if left_dist + right_dist > max_dist:
                    max_dist = left_dist + right_dist
                    furthest_point = current_point
            finger_tips[finger] = furthest_point
            cv2.circle(image_data, furthest_point, 20, [255, 255, 255],
                       -1)
        return (image_data, finger_tips)

    def identifyAlignmentVector(image_data, finger_tips, finger_joins):
        """Calculates and draws the alignment vector that will be used when rotating the hand.
        
        Parameters
        ----------
        image_data : Contains the hand image.
        finger_tips : The coordinates of the finger tips.
        finger_joins : The coordinates of the finger-join defects.
        
        Returns
        -------
        image_data : The hand with the alignment vector drawn on."""

        middle_tip = finger_tips['middle']
        ring_middle = finger_joins['ring-middle']
        middle_index = finger_joins['middle-index']
        mid = (int((ring_middle[0] + middle_index[0]) / 2),
               int((ring_middle[1] + middle_index[1]) / 2))
        try:
            gradient = (middle_tip[1] - mid[1]) / (middle_tip[0]
                    - mid[0])
            c = -(gradient * mid[0]) + mid[1]
            (height, width) = image_data.shape[:2]
            x = math.floor((height - c) / gradient)
        except ZeroDivisionError:
            (height, width) = image_data.shape[:2]
            x = middle_tip[0]
        cv2.line(image_data, middle_tip, (x, height), [255, 255, 255],
                 4)
        return image_data

    def centerHand(image_data, centroid):
        """Centres the hand to with the centre of the image.
        
        Parameters
        ----------
        image_data : Contains the hand image.
        centroid : The centroid of the hands palm.
        
        Returns
        -------
        img_translation : The centres hand."""

        (height, width) = image_data.shape[:2]
        cX = width / 2  # X-coord centroid of image
        cY = height / 2  # Y-coord centroid of image
        (cx, cy) = centroid
        trans_matrix = np.float32([[1, 0, -math.floor(cx - cX)], [0, 1,
                                  -math.floor(cy - cY)]])
        img_translation = cv2.warpAffine(image_data, trans_matrix,
                (width, height))
        return img_translation

    def showAlignmentAccuracy(components):
        """Layers the centred hands ontop of eachother to view the accuracy of alignment.
        
        Parameters
        ----------
        components : The binary hand component.
        
        Returns
        -------
        composition_image : Composition image of layered hands."""

        composition_image = np.zeros((2500, 2500, 3)).astype(np.uint8)

        PHI = (1 + math.sqrt(5)) / 2
        i = 1
        x = lambda component_colour_ID: component_colour_ID * PHI \
            - math.floor(component_colour_ID * PHI)
        initial_colour = (0, 0, 0)
        color = ''
        for component in components:
            color = (initial_colour[0] + x(i ** 2) * 256,
                     initial_colour[1] + x((i + 1) ** 2) * 256,
                     initial_colour[2] + x((i + 2) ** 2) * 256)
            component_colour = cv2.cvtColor(component.astype(np.uint8),
                    cv2.COLOR_GRAY2BGR)
            hsv = cv2.cvtColor(component_colour, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, (0, 0, 1), (255, 255, 255))
            component_colour1 = component_colour.copy()
            component_colour1[mask > 0] = color
            composition_image = cv2.addWeighted(composition_image, 0.8,
                    component_colour1, 0.6, 0)
            i = i + 1
        return composition_image

    def rotateHand(image_data, finger_joins, finger_tips):
        """Rotates the centred hand so that the middle finger points up.
        
        Parameters
        ----------
        image_data : The centred hand data.
        finger_joins : The coordinates of the finger-join defects.
        finger_tips : The coordinates of the finger tips. 
        
        Returns
        -------
        rotated_image : The rotated hand image."""

        tip = finger_tips['middle']
        ring_middle = finger_joins['ring-middle']
        middle_index = finger_joins['middle-index']
        a = np.array([tip[0], tip[1]])
        b = np.array([(ring_middle[0] + middle_index[0]) // 2,
                     (ring_middle[1] + middle_index[1]) // 2])
        c = np.array([middle_index[0], (ring_middle[1]
                     + middle_index[1]) // 2])
        ba = a - b
        bc = c - b
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba)
                                   * np.linalg.norm(bc))
        rotation_angle = np.degrees(np.arccos(cosine))
        (height, width) = image_data.shape[:2]
        cX = width / 2
        cY = height / 2
        M = cv2.getRotationMatrix2D((cX, cY), 90 - rotation_angle, 1.0)
        rotated_image = cv2.warpAffine(image_data, M, (2500, 2500))
        return rotated_image

    def spikyHand(
        image_data,
        finger_joins,
        finger_tips,
        wrist_defects,
        ):
        """Calculates the lenths between the fingertips, finger-joins and wrist defects and packs them into a vector.
        
        Parameters
        ----------
        image_data : Contains the hand image data.
        finger_joins : The coordinates of the finger-join defects.
        finger_tips : The coordinates of the finger tips.
        wrist_defects : The coordinates of the wrist defects. 
        
        Returns
        -------
        image_data : The hand with the feature vectors drawn on.
        hand_similarity_vect : Vector of hand feature lengths"""

        defects = []
        fingertips = []
        defects.append(wrist_defects['wrist-left'])
        defects.append(finger_joins['pinky-ring'])
        defects.append(finger_joins['ring-middle'])
        defects.append(finger_joins['middle-index'])
        defects.append(finger_joins['index-thumb'])
        defects.append(wrist_defects['wrist-right'])
        fingertips.append(finger_tips['pinky'])
        fingertips.append(finger_tips['ring'])
        fingertips.append(finger_tips['middle'])
        fingertips.append(finger_tips['index'])
        fingertips.append(finger_tips['thumb'])

        hand_similarity_vect = []  # Stores the lengths between defect-points and fingertips.
        for i in range(5):
            length = math.sqrt((defects[i][0] - fingertips[i][0]) ** 2
                               + (defects[i][1] - fingertips[i][1])
                               ** 2)
            hand_similarity_vect.append(length)
            length1 = math.sqrt((defects[i + 1][0] - fingertips[i][0])
                                ** 2 + (defects[i + 1][1]
                                - fingertips[i][1]) ** 2)
            hand_similarity_vect.append(length1)

        cv2.line(image_data, defects[0], fingertips[0], [255, 255,
                 255], 4)
        cv2.line(image_data, fingertips[0], defects[1], [255, 255,
                 255], 4)
        cv2.line(image_data, defects[1], fingertips[1], [255, 255,
                 255], 4)
        cv2.line(image_data, fingertips[1], defects[2], [255, 255,
                 255], 4)
        cv2.line(image_data, defects[2], fingertips[2], [255, 255,
                 255], 4)
        cv2.line(image_data, fingertips[2], defects[3], [255, 255,
                 255], 4)
        cv2.line(image_data, defects[3], fingertips[3], [255, 255,
                 255], 4)
        cv2.line(image_data, fingertips[3], defects[4], [255, 255,
                 255], 4)
        cv2.line(image_data, defects[4], fingertips[4], [255, 255,
                 255], 4)
        cv2.line(image_data, fingertips[4], defects[5], [255, 255,
                 255], 4)

        return (image_data, hand_similarity_vect)

    def drawBaseContours(image_data, base_contour):
        """xxxxxx
        
        Parameters
        ----------
        image_data : 
        base_contour : 
        
        Returns
        -------
        image_data : """

        for i in range(len(base_contour)):
            cv2.circle(image_data, base_contour[i][0], 1, [200, 200,
                       200], -1)
        return image_data

    def drawColouredContour(image_data, array_of_square_errors, lines):
        """xxxxxx
        
        Parameters
        ----------
        image_data : 
        array_of_square_errors : 
        lines : 

        Returns
        -------
        image_data : """

        max_displacement = array_of_square_errors[0][1]
        min_displacement = array_of_square_errors[0][1]

        for i in range(1, len(array_of_square_errors)):
            if array_of_square_errors[i][1] > max_displacement:
                max_displacement = array_of_square_errors[i][1]
            elif array_of_square_errors[i][1] < min_displacement:
                min_displacement = array_of_square_errors[i][1]
        range_displacement = max_displacement - min_displacement
        smallest_deviation_colour = (255, 255, 255)
        largest_deviation_colour = (0, 0, 255)
        for i in range(len(array_of_square_errors)):
            func_colour_mapping = (array_of_square_errors[i][1]
                                   - min_displacement) \
                / range_displacement
            temp_colour = [0, 0, 0]
            for j in range(0, 3):
                temp_colour[j] = func_colour_mapping \
                    * (largest_deviation_colour[j]
                       - smallest_deviation_colour[j]) \
                    + smallest_deviation_colour[j]
            current_colour = (temp_colour[0], temp_colour[1],
                              temp_colour[2])
            cv2.circle(image_data, array_of_square_errors[i][0], 5,
                       current_colour, -1)
            if lines == True:
                if i % 20 == 0:
                    cv2.line(image_data, array_of_square_errors[i][0],
                             array_of_square_errors[i][2],
                             current_colour, thickness=2)
        return image_data

    def handComp(hands, ideal_hand):
        """Compares the area and hand boundry lengths of each hand to the ideal hand.
        
        Parameters
        ----------
        hands : Data for all hands in set.
        ideal_hand : Data for ideal hand.
        
        Returns
        -------
        hand_datapoints : List of 2D datapoints."""

        hand_datapoints = []
        (ideal_component, ideal_spiky, ideal_wrist_defects) = ideal_hand  # The binary hand component, hand boundry-length vector and wrist coordinates for the ideal hand.
        (height, width) = ideal_component.shape[:2]
        ideal_centered_component = ideal_component.copy()
        ideal_rotated_component = ideal_component.copy()
        imgCentroidX = width / 2
        imgCentroidY = height / 2
        ideal_wrist_left = ideal_wrist_defects['wrist-left']
        ideal_wrist_right = ideal_wrist_defects['wrist-right']
        def1 = np.array([ideal_wrist_left[0], ideal_wrist_left[1]])
        def2 = np.array([ideal_wrist_right[0], ideal_wrist_right[1]])
        def3 = np.array([def2[0], def1[1]])
        ba = def1 - def2
        bc = def2 - def3
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba)
                                   * np.linalg.norm(bc))
        rotation_angle = np.degrees(np.arccos(cosine))

        # Rotates wristpoints against the horizontal and slices wrist off.

        if def1[1] <= def2[1]:
            M = cv2.getRotationMatrix2D((imgCentroidX, imgCentroidY),
                    rotation_angle - 90, 1.0)
        else:
            M = cv2.getRotationMatrix2D((imgCentroidX, imgCentroidY),
                    90 - rotation_angle, 1.0)
        ideal_rotated_component = \
            cv2.warpAffine(ideal_centered_component, M, (width, height))
        ideal_rotated_component = ideal_rotated_component[:def1[1]
                - height, :]
        analysis = \
            cv2.connectedComponentsWithStats(ideal_rotated_component.astype(np.uint8),
                4, cv2.CV_32S)
        (num_labels, labels, stats, centroid) = analysis
        component_sizes = stats[:, -1]
        ideal_hand_size = component_sizes[1]
        for i in range(1, num_labels):
            if component_sizes[i] > ideal_hand_size:
                ideal_hand_size = component_sizes[i]
        ideal_hand_vector_len = 0
        for i in range(len(ideal_spiky)):  # Summing length of boundries for ideal hand.
            ideal_hand_vector_len = ideal_hand_vector_len \
                + ideal_spiky[i]

        for hand in hands:
            (hand_index, tuple) = hand
            (component, spikyhand, wrist_defects) = tuple  # The binary hand component, hand boundry-length vector and wrist coordinates for each hand in the set.
            (height, width) = component.shape[:2]
            centered_component = component.copy()
            rotated_component = component.copy()
            imgCentroidX = width / 2
            imgCentroidY = height / 2
            wrist_left = wrist_defects['wrist-left']
            wrist_right = wrist_defects['wrist-right']
            def1 = np.array([wrist_left[0], wrist_left[1]])
            def2 = np.array([wrist_right[0], wrist_right[1]])
            def3 = np.array([def2[0], def1[1]])
            ba = def1 - def2
            bc = def2 - def3
            cosine = np.dot(ba, bc) / (np.linalg.norm(ba)
                    * np.linalg.norm(bc))
            rotation_angle = np.degrees(np.arccos(cosine))

            # Rotates wristpoints against the horizontal and slices wrist off.

            if def1[1] <= def2[1]:
                M = cv2.getRotationMatrix2D((imgCentroidX,
                        imgCentroidY), rotation_angle - 90, 1.0)
            else:
                M = cv2.getRotationMatrix2D((imgCentroidX,
                        imgCentroidY), 90 - rotation_angle, 1.0)
            rotated_component = cv2.warpAffine(centered_component, M,
                    (width, height))
            rotated_component = rotated_component[:def1[1] - height, :]
            analysis = \
                cv2.connectedComponentsWithStats(rotated_component.astype(np.uint8),
                    4, cv2.CV_32S)
            (num_labels, labels, stats, centroid) = analysis
            component_sizes = stats[:, -1]
            hand_size = component_sizes[1]
            for i in range(1, num_labels):
                if component_sizes[i] > hand_size:
                    hand_size = component_sizes[i]

            hand_vector_len = 0
            for i in range(len(spikyhand)):  # Summing length of boundries for compared hand.
                hand_vector_len = hand_vector_len + spikyhand[i]

            size_compared = hand_size / ideal_hand_size  # Area similarity as a ratio of compared hand to ideal hand.
            length_compared = hand_vector_len / ideal_hand_vector_len  # Length similarity as a ratio of compared hand to ideal hand.

            hand_datapoints.append((hand_index, (size_compared,
                                   length_compared)))
        hand_datapoints = sorted(hand_datapoints, key=lambda item: \
                                 abs(2 - (item[1][1] + item[1][0])),
                                 reverse=False)

        return hand_datapoints

    def separateContourIntoSegments(
        contour,
        finger_joins,
        wrist_defects,
        finger_tips,
        ):
        """
        Takes in a full contour of the hand and separates it into 10 edges, with each edge between a defect and a fingertip

        Parameters
        -------
        contour: Array of coordinates along the outside of the hand
        finger_joins: Array of finger defect coordinates
        wrist_defect: Array of wrist defect coordinates
        finger_tips: Array of fingertip coordinates

        Returns
        --------
        contour array:An array of smaller contour arrays, with each smaller contour array consisting of an array of points
        between a defect and a fingertip
        """
        defects = []
        fingertips = []
        contour_array = []
        temp_contour = contour
        defects.append(wrist_defects['wrist-right'])
        defects.append(finger_joins['index-thumb'])
        defects.append(finger_joins['middle-index'])
        defects.append(finger_joins['ring-middle'])
        defects.append(finger_joins['pinky-ring'])
        defects.append(wrist_defects['wrist-left'])

        fingertips.append(finger_tips['thumb'])
        fingertips.append(finger_tips['index'])
        fingertips.append(finger_tips['middle'])
        fingertips.append(finger_tips['ring'])
        fingertips.append(finger_tips['pinky'])
        for j in range(len(fingertips)):
            for i in range(len(temp_contour)):

                tuple_contour = (temp_contour[i][0][0],
                                 temp_contour[i][0][1])
                if tuple_contour == defects[j]:
                    outer_edge_start_index = i
                if tuple_contour == (fingertips[j][0],
                        fingertips[j][1]):
                    middle_index = i
                if tuple_contour == defects[j + 1]:
                    end_index = i
            outer_contour = contour[outer_edge_start_index:middle_index]
            inner_contour = contour[middle_index:end_index]
            contour_array.append(outer_contour)
            contour_array.append(inner_contour)
        return contour_array

    def generateAllignmentErrors(base_contour, comparison_contour):
        """
        Takes in an array of base fingeredge arrays, and a corresponding array of comparison hand fingeredge arrays.
        For each point along the contour of comparison contour, finds the point closest to it (however has to be along the same edge (inner/outer))
        of the same finger, ) on the base contour and calculates the distance between them.


        Parameters
        -------
        base_contour: A 3d array – An array consisting of contour arrays of each finger edge of base contour
        comparison_contour: A 3d array – An array consisting of contour arrays of each finger edge of comparison contour


        Returns
        --------
        big_array: A 2d array, with each array consisiting of [point on comparison contour, squared distance, corresponding point on base array] 

    """
        big_array = []
        for j in range(len(base_contour)):

            comparison_contour_size = len(comparison_contour[j])
            base_contour_size = len(base_contour[j])
            current_comparison_edge = comparison_contour[j]
            current_base_edge = base_contour[j]
            edge_array = []
            sum_squared_errors = 0
            for i in range(len(current_comparison_edge)):
                pointOfMinDistance = current_base_edge[0][0]
                minDistance = \
                    math.dist((current_comparison_edge[i][0][0],
                              current_comparison_edge[i][0][1]),
                              (current_base_edge[0][0][0],
                              current_base_edge[0][0][1]))
                for k in range(1, len(current_base_edge)):
                    distance = \
                        math.dist((current_comparison_edge[i][0][0],
                                  current_comparison_edge[i][0][1]),
                                  (current_base_edge[k][0][0],
                                  current_base_edge[k][0][1]))
                    if distance < minDistance:
                        minDistance = distance
                        pointOfMinDistance = current_base_edge[k][0]
                big_array.append([current_comparison_edge[i][0],
                                 minDistance ** 2, pointOfMinDistance])
        return big_array

    def generateContrastArray(base_contour, comparison_contour):
        """
        Takes in an array of base fingeredge arrays, and a corresponding array of comparison hand fingeredge arrays.
        For each point along the contour of comparison contour, finds the point closest to it (however has to be along the same edge (inner/outer))
        of the same finger, ) on the base contour and calculates the distance between them.


        Parameters
        -------
        base_contour: A 3d array – An array consisting of contour arrays of each finger edge of base contour
        comparison_contour: A 3d array – An array consisting of contour arrays of each finger edge of comparison contour


        Returns
        --------
        big_array: A 2d array, with each array consisiting of [point on comparison contour, squared distance, corresponding point on base array] 

        """
        big_array = []

        for j in range(len(base_contour)):
            comparison_smaller = False
            if len(base_contour[j]) < len(comparison_contour[j]):
                smaller_contour = base_contour[j]
                larger_contour = comparison_contour[j]
            else:

                smaller_contour = comparison_contour[j]
                larger_contour = base_contour[j]
                comparison_smaller = True

            ratio = len(smaller_contour) / len(larger_contour)
            for i in range(len(larger_contour)):
                corresponding_pos = int(round(ratio * i))
                point_larger = (larger_contour[i][0][0],
                                larger_contour[i][0][1])
                point_smaller = \
                    (smaller_contour[corresponding_pos][0][0],
                     smaller_contour[corresponding_pos][0][1])
                temp_distance = math.dist(point_smaller, point_larger)
                if comparison_smaller:
                    big_array.append([point_smaller, temp_distance,
                            point_larger])
                else:
                    big_array.append([point_larger, temp_distance,
                            point_smaller])

        return big_array

    def kMeans(data):
        """Creates a k-means cluster from 2D hand data points (boundry lengths and area) as a ratio of the ideal hand.
        
        Parameters
        ----------
        cv_img : 2D data for each hand in the set.
        
        Returns
        -------
        image : Numpy array of clustered data."""

        Y = []
        for i in range(len(data)):
            Y.append(math.sqrt((1 - data[i][0]) ** 2 + (1 - data[i][1])
                     ** 2))
        Z = np.vstack(data)
        Z = np.float32(Z)

        criteria = (cv2.TERM_CRITERIA_EPS, 100, 1.0)
        (ret, label, center) = cv2.kmeans(
            Z,
            1,
            None,
            criteria,
            100,
            cv2.KMEANS_RANDOM_CENTERS,
            )

        A = Z[label.ravel() == 0]

        fig = plt.figure()
        fig.patch.set_facecolor('#0e0e0e')
        fig.patch.set_alpha(0.9)
        ax = fig.add_subplot(111)
        ax.patch.set_facecolor('#212121')
        ax.patch.set_alpha(0.9)
        ax.spines['bottom'].set_color('1')
        ax.spines['top'].set_color('1')
        ax.spines['left'].set_color('1')
        ax.spines['right'].set_color('1')
        ax.xaxis.label.set_color('1')
        ax.tick_params(axis='x', colors='1')
        ax.yaxis.label.set_color('1')
        ax.tick_params(axis='y', colors='1')

        # Colour gets darker as proximity to ideal hand increases.

        plt.scatter(
            A[:, 0],
            A[:, 1],
            s=50,
            c=Y,
            marker='.',
            cmap='BuPu_r',
            )
        plt.scatter(1, 1, s=50, c='red', marker='P')  # Ideal hand.
        plt.scatter(center[:, 0], center[:, 1], s=50, c='y', marker='o')  # Centroid of hand set.
        plt.xlabel('Ratio of Area to Ideal')
        plt.ylabel('Ratio of Boundry Length to Ideal')
        plt.savefig(
            'plot.png',
            dpi=500,
            format=None,
            metadata=None,
            bbox_inches=None,
            pad_inches=0.3,
            facecolor='auto',
            edgecolor='auto',
            backend=None,
            )
        np_scatter = cv2.imread('plot.png', cv2.IMREAD_COLOR)
        return np_scatter

    def saveImages(images, location):
        os.mkdir(location)
        for image in images:
            (name, data) = image
            name = name[:len(name) - 4]
            cv2.imwrite(location + '/' + name + '_aligned' + '.png',
                        data)
