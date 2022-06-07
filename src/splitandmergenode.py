#!/usr/bin/python
# -*- coding: utf-8 -*-

# Basic ROS
import rospy

# ROS messages
from visualization_msgs.msg import Marker
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point

# Maths
import numpy as np

# Custom libraries
from splitandmerge import splitandmerge
from utils_lib.functions import publish_lines, publish_corners

def perp( a ) :
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

# line segment a given by endpoints a1, a2
# line segment b given by endpoints b1, b2
# return 
def seg_intersect(a1,a2, b1,b2) :
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = np.dot( dap, db)
    num = np.dot( dap, dp )
    return np.array([(num / denom.astype(float))*db + b1])

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def line_intersection(a1,a2, b1,b2):
    """find the intersection of line segments A=(a1[0],a1[1])/(a2[0],a2[1]) and
    B=(b1[0],b1[1])/(b2[0],b2[1]). Returns a point or None"""
    denom = ((a1[0] - a2[0]) * (b1[1] - b2[1]) - (a1[1] - a2[1]) * (b1[0] - b2[0]))
    # if denom==0: return None
    px = ((a1[0] * a2[1] - a1[1] * a2[0]) * (b1[0] - b2[0]) - (a1[0] - a2[0]) * (b1[0] * b2[1] - b1[1] * b2[0])) / denom
    py = ((a1[0] * a2[1] - a1[1] * a2[0]) * (b1[1] - b2[1]) - (a1[1] - a2[1]) * (b1[0] * b2[1] - b1[1] * b2[0])) / denom
    if (px - a1[0]) * (px - a2[0]) < 0 and (py - a1[1]) * (py - a2[1]) < 0 \
      and (px - b1[0]) * (px - b2[0]) < 0 and (py - b1[1]) * (py - b2[1]) < 0:
        return np.array([[px, py]])
    else:
        return None

def distance(A, B):
    return np.sqrt((A[1]-B[1])**2 + (A[0]-B[0])**2)

#===============================================================================
class SplitAndMergeNode(object):
    '''
    Class to hold all ROS related transactions to use split and merge algorithm.
    '''
    
    #===========================================================================
    def __init__(self):
        '''
        Initializes publishers and subscribers.
        '''
        # Publishers
        self.pub_map    = rospy.Publisher("map", Marker, queue_size = 2)
        self.pub_line   = rospy.Publisher("lines", Marker,queue_size=0)
        self.pub_laser  = rospy.Publisher("scan_cut",LaserScan,queue_size=0)
        self.pub_corner = rospy.Publisher("corners", Marker, queue_size=2)
        self.pub_corner_gt = rospy.Publisher("corners_gt", Marker, queue_size=2)
        
        # Subscribers
        # self.sub_scan = rospy.Subscriber("scan", LaserScan, self.laser_callback)

        self.sub_scan = rospy.Subscriber("turtlebot/rplidar/scan", LaserScan, self.laser_callback)
        self.split_thres = rospy.get_param("~split_threshold")   # distance threshold to provoke a split
        self.inter_thres = rospy.get_param("~inter_threshold")   # maximum distance between consecutive points in a line
        self.min_points = rospy.get_param("~minimum_points")     # minimum number of points in a line
        self.dist_thres = rospy.get_param("~distance_threshold") # maximum distance to merge lines
        self.ang_thres = np.deg2rad(rospy.get_param("~angle_threshold"))     # maximum angle to merge lines
    
    #===========================================================================
    def laser_callback(self, msg):
        '''S
        Function called each time a LaserScan message with topic "turtlebot/rplidar/scan" arrives. 
        '''
        # Project LaserScan to points in space
        rng = np.array(msg.ranges)
        ang = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        points = np.vstack((rng * np.cos(ang),
                            rng * np.sin(ang))).T
        
        msg.range_max = 3            #Adjusted according to the specs of the RPLidar
        # msg.range_max = 6            #Adjusted according to the specs of the RPLidar

        self.pub_laser.publish(msg)
                            
        # Filter long ranges
        points = points[(rng < msg.range_max) & (rng > msg.range_min), :]
        
        # Return if not enough points available
        if len(points) < 2:
            return
        
        # Use split and merge to obtain lines and publish
        lines = splitandmerge(points, self.split_thres, self.inter_thres, self.min_points, self.dist_thres, self.ang_thres)

        corners = np.array([[0, 0]])
        for j in range(len(lines)-1):
            if distance(lines[j, 2:], lines[j+1, 0:2]) < 0.1 or distance(lines[j, 0:2], lines[j+1, 2:]) < 0.1:
                m1 = (lines[j, 3] - lines[j, 1])/(lines[j,2]-lines[j,0])
                m2 = (lines[j+1, 3] - lines[j+1, 1])/(lines[j+1,2]-lines[j+1,0])
                theta = abs(np.rad2deg(np.arctan2((m1-m2), (1+m1*m2))))

                if theta > 80 and theta < 100 and  distance(lines[j, 0:2], lines[j, 2:]) > 0.2 and distance(lines[j+1, 0:2], lines[j+1, 2:]) > 0.2:
                    corners = np.append(corners, seg_intersect(lines[j,0:2], lines[j,2:], lines[j+1,0:2], lines[j+1,2:]), axis=0)

        publish_corners(corners[1:], self.pub_corner, frame='base_footprint',color=[248/255,188/255,0/255,1], scale=0.25)

        '''Publish results'''
        publish_lines(lines, self.pub_line, frame=msg.header.frame_id,
                      time=msg.header.stamp, ns='scan_line', color=(1,0,1))
 
#===============================================================================       
if __name__ == '__main__':
    
    # ROS initializzation
    rospy.init_node('splitandmerge')
    node = SplitAndMergeNode()
    
    # Continue forever
    rospy.spin()


