#!/usr/bin/python
# -*- coding: utf-8 -*-

"""Main node that connects to the necessary topics."""

# Basic ROS
import rospy
import tf

# ROS messages
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray

# Maths
import numpy as np

# Custom libraries
import probabilistic_lib.functions as funcs
import utils_lib.functions as utilsFunc

# Extended Kalman Filter
from ekf_localization import EKF
from EKFSlamBase import EKFSlam
from PingerWithIDMeasurement import PingerWithIDMeasurement


# ==============================================================================
class LocalizationNode(object):
    """Class to hold all ROS related transactions."""

    # ==========================================================================
    def __init__(self, xinit, odom_lin_sigma, odom_ang_sigma, meas_rng_noise,
                 meas_ang_noise, rob2sensor):
        """Initialize publishers, subscribers and the filter."""
        # Publishers
        self.pub_laser = rospy.Publisher("ekf_laser", LaserScan, queue_size=2)
        self.pub_lines = rospy.Publisher("linesekf", Marker, queue_size=2)
        self.pub_corners = rospy.Publisher("cornersekf", Marker, queue_size=2)
        self.pub_odom = rospy.Publisher("predicted_odom", Odometry,
                                        queue_size=2)
        self.pub_odom_prediction = rospy.Publisher("odom_prediction", Odometry,
                                        queue_size=2)
        self.pub_uncertainity = rospy.Publisher("uncertainity", Marker,
                                                queue_size=2)
        self.pub_corner_uncer = rospy.Publisher("crn_uncertainity", MarkerArray, queue_size=2)

        # Subscribers
        # self.sub_laser = rospy.Subscriber("scan", LaserScan, self.cbk_laser)
        self.sub_corners = rospy.Subscriber("corners", Marker, self.cbk_corners)
        self.sub_odom = rospy.Subscriber("turtlebot/odom", Odometry, self.cbk_odom)
        # self.sub_odom = rospy.Subscriber("odom", Odometry, self.cbk_odom)

        # TF
        self.tfBroad = tf.TransformBroadcaster()

        # Incremental odometry
        self.last_odom = None
        self.odom = None

        # Times
        self.time = rospy.Time(0)
        self.odomtime = rospy.Time(0)
        self.linestime = rospy.Time(0)

        # Flags
        self.uk = None
        self.lines = None
        self.new_odom = False
        self.new_laser = False
        self.pub = False

        self.Qk = np.array([[odom_lin_sigma**2, 0, 0],
                            [0, odom_lin_sigma**2, 0],
                            [0, 0, odom_ang_sigma**2]])

        # Filter
        self.ekf = EKF(xinit, odom_lin_sigma, odom_ang_sigma, meas_rng_noise,
                       meas_ang_noise)

        P_init = np.eye(3)*0.0001
        self.ekf_slam = EKFSlam(xinit, P_init, 100, 2, None)

        # Transformation from robot to sensor
        self.robot2sensor = np.array(rob2sensor)
        # print(self.robot2sensor)

    # ==========================================================================
    def cbk_laser(self, msg):
        """Republish laser scan in the EKF solution frame."""
        msg.header.frame_id = 'sensor'
        self.pub_laser.publish(msg)

    # ==========================================================================
    def cbk_odom(self, msg):
        """Publish tf and calculate incremental odometry."""
        # Save time
        print('-- ODOM CALLBACK')
        self.odomtime = msg.header.stamp
        self.odom = msg

        # Translation
        trans = (msg.pose.pose.position.x,
                 msg.pose.pose.position.y,
                 msg.pose.pose.position.z)

        # Rotation
        rot = (msg.pose.pose.orientation.x,
               msg.pose.pose.orientation.y,
               msg.pose.pose.orientation.z,
               msg.pose.pose.orientation.w)

        # Publish transform
        self.tfBroad.sendTransform(translation=self.robot2sensor,
                                   rotation=(0, 0, 0, 1),
                                   time=msg.header.stamp,
                                   child='sensor',
                                   parent='robot')
        self.tfBroad.sendTransform(translation=self.robot2sensor,
                                   rotation=(0, 0, 0, 1),
                                   time=msg.header.stamp,
                                   child='camera_depth_frame',
                                   parent='base_footprint')
        self.tfBroad.sendTransform(translation=trans,
                                   rotation=rot,
                                   time=msg.header.stamp,
                                   child='base_footprint',
                                   parent='world')
        self.tfBroad.sendTransform(translation=(0, 0, 0),
                                   rotation=(0, 0, 0, 1),
                                   time=msg.header.stamp,
                                   child='odom',
                                   parent='world')

        # Incremental odometry
        if self.last_odom is not None:
            print('INSIDE IF')

            # Increment computation
            delta_x = msg.pose.pose.position.x - \
                self.last_odom.pose.pose.position.x
            delta_y = msg.pose.pose.position.y - \
                self.last_odom.pose.pose.position.y
            yaw = funcs.yaw_from_quaternion(msg.pose.pose.orientation)
            lyaw = funcs.yaw_from_quaternion(
                self.last_odom.pose.pose.orientation)

            # Odometry seen from vehicle frame
            self.uk = np.array([delta_x * np.cos(lyaw) +
                                delta_y * np.sin(lyaw),
                                -delta_x * np.sin(lyaw) +
                                delta_y * np.cos(lyaw),
                                funcs.angle_wrap(yaw - lyaw)])

            # Flag available
            self.new_odom = True

        # Save initial odometry for increment
        else:
            self.last_odom = msg

    # ==========================================================================
    def cbk_corners(self, msg):
        """Republish the laser scam in the /robot frame."""
        # Save time
        self.linestime = msg.header.stamp

        print('--- CORNERS SUBSCRIBE', msg.points)

        # Get the lines
        if len(msg.points) > 0:

            # Structure for the lines
            self.lines = np.zeros((len(msg.points), 2))

            for i in range(0, len(msg.points)):
                # Get start and end points
                pt1 = msg.points[i]

                # Transform to robot frame
                pt1R = funcs.comp(self.robot2sensor, [pt1.x, pt1.y, 0.0])

                # Append to line list
                # self.lines[i, :2] = pt1R[:2]
                self.lines[i, :2] = [msg.points[i].x, msg.points[i].y] 

            # Flag
            self.new_laser = True

            # Publish
            # funcs.publish_lines(self.lines, self.pub_lines, frame='robot',
            #                     time=msg.header.stamp, ns='lines_robot',
            #                     color=(0, 0, 1))
            print('----- CORNERS', self.lines)
            # utilsFunc.publish_corners(self.lines, self.pub_corners, frame='robot', scale=0.3)

    # ==========================================================================
    def iterate(self):
        """Main loop of the filter."""
        # Prediction
        xr, Pr, Prm = None, None, None
        if self.new_odom:
            # Make prediction (copy needed to ensure no paral thread)
            self.ekf.predict(self.uk.copy())
            print('---- self.uk', self.uk)
            [xr, Pr, Prm] = self.ekf_slam.prediction(self.uk.copy(), self.Qk, 1.0)
            self.ekf_slam.applyPrediction(xr, Pr, Prm)
            self.last_odom = self.odom  # new start odom for incremental
            self.new_odom = False
            self.pub = True
            self.time = self.odomtime
            odom = funcs.get_odom_msg(self.ekf.xk)
            self.pub_odom_prediction.publish(odom)

        # Data association and update
        if self.new_laser:
            # Make data association and update
            temp = self.ekf_slam.is_data_associated(self.lines.copy())
            # temp = self.ekf.data_association(self.lines.copy())
            associd, distance = temp
            for index, corner in enumerate(self.lines):
                # if corner[0] == 0 and corner[1] == 0:
                #     print('HERE123')
                #     continue
                if index in associd.keys(): #Known landmark! Do Upate
                    idx = associd[index]
                    xr = self.ekf_slam.xr()
                    xl = self.ekf_slam.xl(idx)
                    [x, P] = self.ekf_slam.update(PingerWithIDMeasurement.h([0,0,0], corner),
                                        self.ekf_slam.Rk,
                                        idx,
                                        PingerWithIDMeasurement.h(xr, xl), 
                                        PingerWithIDMeasurement.Jhxr(xr, xl),
                                        PingerWithIDMeasurement.Jhxl(xr, xl),
                                        PingerWithIDMeasurement.Jhv())
                    print("-----, APPLY UPDATE", idx, index)
                    self.ekf_slam.applyUpdate(x, P)
                elif distance[index] > 1.0: #Unknown landmark! Do initialization
                # else:
                    idx = self.ekf_slam.add_landmark(PingerWithIDMeasurement.g(self.ekf_slam.xr(), PingerWithIDMeasurement.h([0,0,0], corner)),
                                        PingerWithIDMeasurement.Jgxr(self.ekf_slam.xr(), PingerWithIDMeasurement.h([0,0,0], corner)),
                                        PingerWithIDMeasurement.Jgz(self.ekf_slam.xr(), PingerWithIDMeasurement.h([0,0,0], corner)), 
                                        self.ekf_slam.Rk)
                    print("-----, ADD LANDMARK")

            # self.ekf.update_position(Hk_list, Vk_list, Sk_list, Rk_list)
            self.new_laser = False
            self.pub = True
            self.time = self.linestime

        # Publish results
        if self.pub:
            self.publish_results()
            self.pub = False

    # ==========================================================================
    def publish_results(self):
        """Publishe results from the filter."""
        # Map of the room (ground truth)
        funcs.publish_lines(self.ekf.map, self.pub_lines, frame='world',
                            ns='map', color=(0, 1, 0))

        utilsFunc.publish_corners((self.ekf_slam.x()[3:]).reshape(self.ekf_slam.n_landmarks_, 2), self.pub_corners, frame='world', 
                            scale=0.3, color=(1, 1, 0, 1))
        
        print('NUMBER OF LANDMARKS', self.ekf_slam.n_landmarks_, self.ekf_slam.Prr())

        # Get filter data
        # odom, ellipse, trans, rot, dummy = funcs.get_ekf_msgs(self.ekf)

        odom, ellipse, trans, rot, dummy = funcs.get_ekf_slam_msgs(self.ekf_slam)
        funcs.publish_arrays(self.ekf_slam.Pmm(), self.pub_corner_uncer, self.ekf_slam)

        # print('----', ellipse)

        # Publish results
        self.pub_odom.publish(odom)
        self.pub_uncertainity.publish(ellipse)
        self.tfBroad.sendTransform(translation=trans,
                                   rotation=rot,
                                   time=self.time,
                                   child='robot',
                                   parent='world')

# ======================================================================
if __name__ == '__main__':

    # ROS initializzation
    rospy.init_node('localization')
    node = LocalizationNode(xinit=[0.0, 0.0, 0.0],
                            odom_lin_sigma=0.025,
                            odom_ang_sigma=np.deg2rad(2),
                            meas_rng_noise=0.2,
                            meas_ang_noise=np.deg2rad(10),
                            rob2sensor=[0.0, 0.0, np.deg2rad(0)])
    # Filter at 10 Hz
    r = rospy.Rate(10)
    while not rospy.is_shutdown():
        # Iterate filter
        node.iterate()
        r.sleep()
