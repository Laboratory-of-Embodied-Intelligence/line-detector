#!/usr/bin/env python2.7

import rospy
from sensor_msgs.msg import Image
import numpy as np
import cv2
from cv_bridge import CvBridge
from scipy import interpolate as inter
from copy import deepcopy as dp

IMAGE_TOPIC = "/rgb_image"
M = np.array([[1.00792, 2.38847, -10.5], [0, 4.74203, 3], [0, 0.00419, 1]  ])
M_INV = np.array([[1, -0.5411, 12.041], [0, 0.22424, -0.67272], [0, -0.000936, 1.0028]  ])
IMG_SIZE = (640, 330)

def get_rectangles(contours, img):

    font = cv2.FONT_HERSHEY_SIMPLEX

    if len(contours) == 0:
        return

    centroids = []
    for cnt in contours:
        S = cv2.contourArea(cnt)
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            M["m00"] += 1
        center_x = int( M["m10"] / M["m00"] )
        center_y = int( M["m01"] / M["m00"] )
        centroids.append([center_x, center_y])

    if len(centroids) == 0:
        return

    a_points = []
    d_a = []
    D_points = []
    d_m = []
    for i in range( len(contours) ):
        min_dist = np.inf
        max_dist = -1
        center = centroids[i]
        for point in contours[i]:
            dist = np.sqrt( abs( (point[0][0] - center[0])**2 + (point[0][1] - center[1])**2 ) )
            if dist < min_dist:
                min_dist = dist
                a = point[0]
            if dist > max_dist:
                max_dist = dist
                D = point[0]
        a_points.append(a)
        d_a.append(min_dist)
        D_points.append(D)
        d_m.append(max_dist)


    b_points = []
    d_b = []
    for i in range( len(contours) ):
        center = centroids[i]
        slope_fixed = ( a_points[i][1] - center[1] ) / ( a_points[i][0] - center[0] )
        min_val = np.inf
        for point in contours[i]:
            slope = ( point[0][1] - center[1] ) / ( point[0][0] - center[0] )
            val = abs( slope * slope_fixed + 1 )
            if val < min_val:
                min_val = val
                b = point[0]
        b_points.append(b)
        b_dist = np.sqrt( abs( (b[0] - center[0])**2 + (b[1] - center[1])**2 ) )
        d_b.append(b_dist)

    new_contours = []

    for i in range( len(contours) ):
        S = cv2.contourArea(contours[i])
        r_s = abs( (S - d_a[i]*d_b[i])/S )
        r_l = abs( (d_m[i] - np.sqrt( d_a[i]**2 + d_b[i]**2 )/d_m[i] ) )
        #if r_s > 0.7 and r_s < 0.9 and r_l > 24 and r_l < 50 :
        new_contours.append(contours[i])
        """
        M = cv2.moments(contours[i])
        if M["m00"] == 0:
            M["m00"] += 1
        center_x = int( M["m10"] / M["m00"] )
        center_y = int( M["m01"] / M["m00"] )
        #rospy.loginfo("r_s = {0:.5f};  r_l = {1:.5f}".format(r_s, r_l ))
        msg = "{:.4f};  {:.4f}".format(r_s, r_l)
        cv2.putText(img, msg, (center_x,center_y), font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        """

    return np.array(new_contours), img

def hist_eq(img, alpha = 4.0, kernel = (5, 5)):

    ### Lightness correction algo ###
    # https://web-answers.ru/c/prostaja-korrekcija-osveshhennosti-v.html #

    img = img[150:,:,:]

    #-----Converting image to LAB Color model-----------------------------------
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    #-----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)

    #-----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=alpha, tileGridSize=kernel)
    cl = clahe.apply(l)

    #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl,a,b))

    return limg

def get_rectangles_ratio(img):
    size = (img.shape[1], img.shape[0])
    #img = cv2.warpPerspective(img, M, IMG_SIZE, flags=cv2.INTER_LINEAR)
    L, a, b = cv2.split(img)

    _, thresh_b = cv2.threshold(b, 180, 255, cv2.THRESH_BINARY)
    _, thresh_L = cv2.threshold(L, 200, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3,20),np.uint8)
    closing_b = cv2.morphologyEx(thresh_b, cv2.MORPH_OPEN, kernel)
    closing_b = cv2.morphologyEx(closing_b, cv2.MORPH_OPEN, kernel)

    closing_L = cv2.morphologyEx(thresh_L, cv2.MORPH_OPEN, kernel)
    closing_L = cv2.morphologyEx(closing_L, cv2.MORPH_OPEN, kernel)

    _, contours, hierarchy = cv2.findContours(cv2.bitwise_and( thresh_L, thresh_b ), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    _, contours, hierarchy = cv2.findContours(thresh_L, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        perimeter = cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt, 0.1 * perimeter, True)
        #rospy.loginfo("Len Approx: {}".format(len(approx)), )
        if len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            area = cv2.contourArea(cnt)
            #rospy.loginfo("Ar: {}".format(ar), )
            if (ar >= 1.8 or ar <= 1.2) and (area > 50 and area < 8500):
                cv2.drawContours(img,[cnt],0,(255,255,255),-1)

    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    #img = cv2.warpPerspective(img, M_INV, IMG_SIZE)
    #trans_matrix = np.float32( [[1, 0, 0], [0, 1, 100]] )
    #img = cv2.warpAffine( img, trans_matrix, IMG_SIZE )
    return img


def get_rectangles_area(img):
    size = (img.shape[1], img.shape[0])
    img = cv2.warpPerspective(img, M, size, flags=cv2.INTER_NEAREST)
    L, a, b = cv2.split(img)

    _, thresh_b = cv2.threshold(b, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    _, thresh_L = cv2.threshold(L, 200, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3,20),np.uint8)
    closing_b = cv2.morphologyEx(thresh_b, cv2.MORPH_OPEN, kernel)
    closing_b = cv2.morphologyEx(closing_b, cv2.MORPH_OPEN, kernel)

    closing_L = cv2.morphologyEx(thresh_L, cv2.MORPH_OPEN, kernel)
    closing_L = cv2.morphologyEx(closing_L, cv2.MORPH_OPEN, kernel)

    _, contours, hierarchy = cv2.findContours(cv2.bitwise_and( thresh_L, thresh_b ), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #_, contours, hierarchy = cv2.findContours( thresh_b, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        area = cv2.contourArea(box)
        if area > 50 and area < 8500:
            cv2.drawContours(img, [cnt], -1, (0,255,0), -1)

    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    #border = np.full((330, 640, 3), 0).astype(np.uint8)
    #rospy.loginfo("Size border: {}".format(border.shape), )
    #rospy.loginfo("Size image : {}".format(img.shape), )
    #img = np.concatenate( (border, img), axis=0 )
    #img = cv2.warpPerspective(img, M_INV, IMG_SIZE)

    return img

def get_RGB_image(msg):

    img = bridge.imgmsg_to_cv2(msg)
    #img_lab = hist_eq(img, alpha = 5.0)
    img = cv2.GaussianBlur(img,(5,5),0)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(img_lab)
    #segmented_img_ratio = get_rectangles_ratio(dp(img_lab))
    #segmented_img_area = get_rectangles_area(dp(img_lab))

    #new_segm_msg_ratio = bridge.cv2_to_imgmsg( segmented_img_ratio, "bgr8")
    #new_segm_msg_area = bridge.cv2_to_imgmsg( segmented_img_area, "bgr8")

    _, thresh_b = cv2.threshold(b, 140, 255, cv2.THRESH_BINARY)

    ch_L = bridge.cv2_to_imgmsg( L, "mono8")
    ch_b = bridge.cv2_to_imgmsg( thresh_b, "mono8")
    ch_a = bridge.cv2_to_imgmsg( a, "mono8")

    #yellow_line_ratio.publish(new_segm_msg_ratio)
    #yellow_line_area.publish(new_segm_msg_area)

    channel_L.publish(ch_L)
    channel_b.publish(ch_b)
    channel_a.publish(ch_a)


if __name__ == '__main__':
    bridge = CvBridge()

    rospy.init_node("rgb_to_cie", log_level=rospy.INFO)
    #yellow_line_ratio = rospy.Publisher("yellow_line_ratio", Image, queue_size=10)
    #yellow_line_area = rospy.Publisher("yellow_line_area", Image, queue_size=10)
    #yellow_line_area = rospy.Publisher("yellow_line_area", Image, queue_size=10)

    channel_L = rospy.Publisher("channel_L", Image, queue_size=10)
    channel_b = rospy.Publisher("channel_b", Image, queue_size=10)
    channel_a = rospy.Publisher("channel_a", Image, queue_size=10)

    rospy.Subscriber(IMAGE_TOPIC, Image, get_RGB_image)

    while not rospy.is_shutdown():
        rospy.spin()
