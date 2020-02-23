# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 14:32:11 2020

@author: Prakash
"""

import cv2
import numpy as np
import os
import cv2.aruco as aruco
import sys 
codes_folder_path = os.path.abspath('.')
images_folder_path = os.path.abspath(os.path.join('..', 'Images'))
op_images_folder_path = os.path.abspath(os.path.join('..', 'Op_Images'))
crop_images_folder_path=os.path.abspath(os.path.join('..', 'Cropped_Images'))

def image_capture():
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    ret, frame = cap.read()
    
    i=1
    
    while(ret):
        ret, frame = cap.read()
        # display to see if the frame is correct
        cv2.imshow("window", frame)
        
        #img= cv2.fastNlMeansDenoisingColored(frame,None,10,10,7,21)
        img=frame
        cv2.imwrite(images_folder_path+"/"+str(i)+".jpg",img)
        k = cv2.waitKey(int(1000/fps)) & 0xff
        i=i+1
        # calling the algorithm function
        #op_image = process(frame)
        #cv2.imshow('image', op_image)
        #cv2.imwrite(op_images_folder_path+"/"+str(i)+".jpg",op_image)
        if k == 27:
            break
    return images_folder_path

def detect_Aruco(img):
    aruco_list = {}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_50)  
    parameters = aruco.DetectorParameters_create() 
    
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters = parameters)
    
    #gray = aruco.drawDetectedMarkers(gray, corners,ids)
    # cv2.imshow('frame',gray)
    #print (type(corners[0]))
    if len(corners):
        #print (len(corners))
        #print (len(ids))
        for k in range(len(corners)):
            temp_1 = corners[k]
            temp_1 = temp_1[0]
            temp_2 = ids[k]
            temp_2 = temp_2[0]
            aruco_list[temp_2] = temp_1
        return aruco_list

def Aruco_centre(img, aruco_list):
    key_list = aruco_list.keys()
    for key in key_list:
        dict_entry = aruco_list[key]
        centre = dict_entry[0] + dict_entry[1] + dict_entry[2] + dict_entry[3]
        centre[:] = [int(x / 4) for x in centre]    
        #print (centre)
        orient_centre = centre + [0.0,5.0]
        #print (orient_centre)
        centre = tuple(centre)  
        orient_centre = tuple((dict_entry[0]+dict_entry[1])/2)
        #print (centre)
        #print orient_centre
        cv2.circle(img,centre,1,(0,0,255),8)
        cv2.circle(img,tuple(dict_entry[0]),1,(0,0,255),8)
        cv2.circle(img,tuple(dict_entry[1]),1,(0,255,0),8)
        cv2.circle(img,tuple(dict_entry[2]),1,(255,0,0),8)
        cv2.circle(img,orient_centre,1,(0,0,255),8)
        cv2.line(img,centre,orient_centre,(255,0,0),4)
    return centre

def image_crop(img,m,n):
    copy=img.copy();
    w=copy.shape[0]
    h=copy.shape[1]
    
    for i in range(0,h,m):
        for j in range(0,w,n):
            crop=copy[w:w+n,h:h+m]
            cv2.imwrite(crop_images_folder_path+'\\'+str(i)+str(j)+'.jpg',crop)
    return crop_images_folder_path
'''
class Node():

    def __init__(self, par=None, pos=None):
        self.par = par
        self.pos = pos

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.pos == other.pos
'''
# a funcion to finf shortest path from bot to location with high plastic using a* algo
class Node:
    """
    A node class for A* Pathfinding
    """

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position


def return_path(current_node):
    path = []
    current = current_node
    while current is not None:
        path.append(current.position)
        current = current.parent
    return path[::-1]


def astar(Mat, start, end):

    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    open_list = []
    closed_list = []

    open_list.append(start_node)

    outer_iterations = 0
    max_iterations = (len(Mat) // 2) ** 2

    adjacent_squares = ((0, -1), (0, 1), (-1, 0), (1, 0),)

    while len(open_list) > 0:
        outer_iterations += 1

        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index
                
        if outer_iterations > max_iterations:

            print("giving up on pathfinding too many iterations")
            return return_path(current_node)

        open_list.pop(current_index)
        closed_list.append(current_node)

        if current_node == end_node:
            return return_path(current_node)

        children = []
        
        for new_position in adjacent_squares:  

            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            within_range_criteria = [
                node_position[0] > (len(Mat) - 1),
                node_position[0] < 0,
                node_position[1] > (len(Mat[len(Mat) - 1]) - 1),
                node_position[1] < 0,
            ]
            
            if any(within_range_criteria):
                continue

            if Mat[node_position[0]][node_position[1]] != 0:
                continue

            new_node = Node(current_node, node_position)

            children.append(new_node)
        for child in children:
            if len([closed_child for closed_child in closed_list if closed_child == child]) > 0:
                continue
            child.g = current_node.g + 1
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + \
                      ((child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g + child.h
            if len([open_node for open_node in open_list if child == open_node and child.g > open_node.g]) > 0:
                continue
            open_list.append(child)
# a function to dtermine movement instrns of bot based on current location and path of the bot
def Bot_movement(path,img):
    aruco_list=detect_Aruco(img)
    key=path.top()
    Bot_pos=Aruco_centre(img,aruco_list)
    if Bot_pos[0] in range(0.9*key[0],1.1*key[0]) and Bot_pos[1] in range(0.9*key[1],1.1*key[1]) :
        path.pop()
        mov='s'
        return path,mov
    elif Bot_pos[0] in range(0.9*key[0],1.1*key[0]) and Bot_pos[1]>key[1]:
        mov='d'
        return path,mov
    elif Bot_pos[1] in range(0.9*key[1],1.1*key[1]) and Bot_pos[0]>key[0]:
        mov='l'
        return path,mov
    elif Bot_pos[1] in range(0.9*key[1],1.1*key[1]) and Bot_pos[0]<key[0]:
        mov='r'
        return path,mov
    elif Bot_pos[0] in range(0.9*key[0],1.1*key[0]) and Bot_pos[1]<key[1]:
        mov='u'
        return path,mov

        
        
        
        
    
    
    
         
    