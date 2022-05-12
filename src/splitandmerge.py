#!/usr/bin/python
# -*- coding: utf-8 -*-
# Hands-on Localization Project
# Muhammad Umar
# Changoluisa Ivan
# Nafees Bin Zaman
import numpy as np
from numpy.core.fromnumeric import _nonzero_dispatcher
from utils_lib.functions import angle_wrap
import math

#===============================================================================
def splitandmerge(points, split_thres=0.1, inter_thres=0.3, min_points=6, dist_thres=0.12, ang_thres=np.deg2rad(10)):
    '''
    Takes an array of points in shape (N, 2) being N the number of points, and
    the columns the point in the form [x, y].

    Returns an array of lines of shape (L, 4) being L the number of lines, and
    the columns the initial and final point of each line in the form
    [x1 y1 x2 y2].

    split_thres: distance threshold to provoke a split
    inter_thres: maximum distance between consecutive points in a line
    min_point  : minimum number of points in a line
    dist_thres : maximum distance to merge lines
    ang_thres  : maximum angle to merge lines
    '''
    lines = split(points, split_thres, inter_thres, min_points, 0, len(points)-1)
    return merge(lines, dist_thres, ang_thres)

#===============================================================================
def split(points, split_thres, inter_thres, min_points, first_pt, last_pt):
    '''
    Find lines in the points provided.
    first_pt: column position of the first point in the array
    last_pt : column position of the last point in the array
    '''
    assert first_pt >= 0
    assert last_pt <= len(points) - 1

    # Check minimum number of points
    if len(points) < min_points:
        return None

    # Line defined as "a*x + b*y + c = 0"
    # extracting initial and last point of the given set. 
    x1 = points[first_pt, 0]
    y1 = points[first_pt, 1]
    x2 = points[last_pt,  0]
    y2 = points[last_pt,  1]
    #computing (a,b,c) line parameters. 
    a = y1-y2
    b = x2-x1
    c = x1*y2-x2*y1
    max_distance = 0
    max_dist_ind = last_pt
    #Distances of points to line 
    for i in range(first_pt, last_pt+1):
        distance = abs(a*points[i,0]+b*points[i,1]+c)/math.sqrt(a**2+b**2)
        if distance > max_distance:
            max_distance = distance
            #save the index in which the condition was true 
            max_dist_ind = i

    if max_distance > split_thres:
        #split the points in two subsets and call the split() function 
        prev = split(points[first_pt:max_dist_ind,:], split_thres, inter_thres, min_points, 0, len(points[first_pt:max_dist_ind,:])-1)
        post = split(points[(max_dist_ind+1):last_pt,:], split_thres, inter_thres, min_points, 0, len(points[(max_dist_ind+1):last_pt,:])-1)

        # Return results of sublines
        if prev is not None and post is not None:
            return np.vstack((prev, post))
        elif prev is not None:
            return prev
        elif post is not None:
            return post
        else:
            return None
    else:
        #Check distance between consecutive points 
        for j in range(first_pt, last_pt-1):
            interdist = math.sqrt((points[j+1,0]-points[j,0])**2 + (points[j+1,1]-points[j,1])**2)
            #If the distance is greater than the threshold, split the points in 2 subsets. 
            if interdist > inter_thres:
                prev = split(points[first_pt:j,:], split_thres, inter_thres, min_points, 0, len(points[first_pt:j,:])-1)
                post = split(points[(j+1):last_pt,:], split_thres, inter_thres, min_points, 0, len(points[(j+1):last_pt,:])-1)
                if prev is not None and post is not None:
                    return np.vstack((prev, post))
                elif prev is not None:
                    return prev
                elif post is not None:
                    return post
                else:
                    return None
    #Return original line if none of the conditions was true 
    return np.array([[x1, y1, x2, y2]])

#===============================================================================
def merge(lines, dist_thres, ang_thres):
    '''
    Merge similar lines according to the given thresholds.
    '''
    # No data received
    if lines is None:
        return np.array([])

    # Check and merge similar consecutive lines
    i = 0
    while i in range(len(lines)-1):

        # Line angles
        #Computing the angles for each line and its consecutive
        ang1 = np.arctan2((lines[i,3] - lines[i,1]), (lines[i,2] - lines[i,0]))
        ang2 = np.arctan2((lines[i+1,3] - lines[i+1,1]), (lines[i+1,2] - lines[i+1,0]))
        # Below thresholds?
        angdiff = abs(angle_wrap(ang1-ang2))

        #computing the distance between the last point of the first line and the first point
        #of the consecutive line
        disdiff = math.sqrt((lines[i,2]-lines[i+1,0])**2+(lines[i,3]-lines[i+1,1])**2)

        if angdiff < ang_thres and disdiff < dist_thres:
            lines[i,:] = [lines[i,0], lines[i,1], lines[i+1,2], lines[i+1,3]]
            #delete unnecesary lines
            lines = np.delete(lines, i+1, 0)
        else:
            i += 1
    return lines
