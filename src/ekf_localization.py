#!/usr/bin/python
# -*- coding: utf-8 -*-

"""EKF class that Implements prediction and update."""

import numpy as np
import math
import probabilistic_lib.functions as funcs

# use: comp, get_polar_line, get_map
#Authors:
#Muhammad Umar
#Changoluisa Iván
#Nafees Bin Zaman

# ==============================================================================
class EKF(object):
    """Class to hold the whole Extended Kalman Filter (EKF)."""

    # ==========================================================================
    def __init__(self, xinit, odom_lin_sigma, odom_ang_sigma, meas_rng_noise,
                 meas_ang_noise):
        """
        Initialize the EKF filter.
        

        Input:
          room_map : a nx4 array of lines in the form [x1 y1 x2 y2]
          xinit    : initial position
          odom_lin_sigma: odometry linear noise
          odom_ang_sigma: odometry angular noise
          meas_rng_noise: measurement linear noise
          meas_ang_noise: measurement angular noise
        """
        # Map with initial displacement
        self.map = funcs.get_dataset3_map(xinit[0], xinit[1], xinit[2])

        # Prediction noise
        self.Qk = np.array([[odom_lin_sigma**2, 0, 0],
                            [0, odom_lin_sigma**2, 0],
                            [0, 0, odom_ang_sigma**2]])

        # Measurement noise
        self.Rk = np.array([[meas_rng_noise**2, 0],
                            [0, meas_ang_noise**2]])

        # Pose initialization
        self.xk = np.zeros(3)
        self.Pk = 0.2 * 0.2 * np.eye(3)  # initial uncertainty of robot state

    # ==========================================================================
    def predict(self, uk):
        """
        Implement the prediction equations of the EKF.

        Saves the new robot state and uncertainty.

        Input:
          uk : numpy array [shape (3,) with (dx, dy, dtheta)]
        """  

        #Jacobian matrices A and W to estimate the predicted covariance 
        #A : Jacobian of the motion model with respect to the state 
        Ak = np.array([[1,0, -np.sin(self.xk[2])*uk[0]-np.cos(self.xk[2])*uk[1]],[0,1,np.cos(self.xk[2])*uk[0]-np.sin(self.xk[2])*uk[1]],[0,0,1]])
        #W : Jacobian of the motion model with respect to the noise 
        Wk = np.array([[np.cos(self.xk[2]), -np.sin(self.xk[2]),0],[np.sin(self.xk[2]),np.cos(self.xk[2]),0],[0,0,1]])
        #Since the covariance is evaluated in the previous state, it has to be computed before predicting the current state 
        self.Pk = Ak @ self.Pk @ np.transpose(Ak) + Wk @ self.Qk @ np.transpose(Wk)

        #Prediction of the current state defined as the compound of the previous state with the most recent control action uk 
        self.xk = funcs.comp(self.xk,uk)
        
        

    # ==========================================================================
    def data_association(self, lines):
        """
        Look for closest correspondences.

        The correspondences are between the provided measured lines and the map
        known a priori.

        Input:
          lines : nx4 matrix with a segment in each row as [x1 y1 x2 y2]
        Return:
          Hk_list : list of 2x3 matrices (jacobian)
          Yk_list : list of 2x1 matrices (innovation)
          Sk_list : list of 2x2 matrices (innovation uncertainty)
          Rk_list : list of 2x2 matrices (measurement uncertainty)
        """

        # Init variables
        chi_thres = 0.103  # chi square 2DoF 95% confidence
        associd = list()
        Hk_list = list()
        Vk_list = list()
        Sk_list = list()
        Rk_list = list()


        # For each observed line
        print('\n-------- Associations --------')
        for i in range(0, lines.shape[0]):

            # The polar lines are already in the robot frame 
            z = funcs.get_polar_line(lines[i],[0,0,0])

            # Variables for finding minimum
            minD = 1e9
            minj = -1

            # For each line in the known map
            for j in range(0, self.map.shape[0]):

                # Compute matrices
                #The Jacobian needs the polar line of the map in the world frame
                h = funcs.get_polar_line(self.map[j],[0,0,0]) 
                H = self.jacobianH(h,self.xk)
                #Then for the innovation, the map line has to be in the robot frame
                h = funcs.get_polar_line(self.map[j],self.xk)
                v = z-h
                S = H @ self.Pk @ np.transpose(H) + self.Rk

                # Mahalanobis distance
                D = np.transpose(v)@ np.linalg.inv(S) @v

                # Optional: Check if observed line is longer than map
                ########################################################
                islonger = False

                # Check if the obseved line is the one with smallest
                # mahalanobis distance
                if np.sqrt(D) < minD and not islonger:
                     minj = j
                     minz = z
                     minh = h
                     minH = H
                     minv = v
                     minS = S
                     minD = D

            # Minimum distance below threshold
            if minD < chi_thres:
                 print("\t{} -> {}".format(minz, minh))
                 # Append results
                 associd.append([i, minj])
                 Hk_list.append(minH)
                 Vk_list.append(minv)
                 Sk_list.append(minS)
                 Rk_list.append(self.Rk)

        return Hk_list, Vk_list, Sk_list, Rk_list


    def is_data_associated(self, lines):
        """
        Look for closest correspondences.

        The correspondences are between the provided measured lines and the map
        known a priori.

        Input:
          lines : nx4 matrix with a segment in each row as [x1 y1 x2 y2]
        Return:
          Hk_list : list of 2x3 matrices (jacobian)
          Yk_list : list of 2x1 matrices (innovation)
          Sk_list : list of 2x2 matrices (innovation uncertainty)
          Rk_list : list of 2x2 matrices (measurement uncertainty)
        """

        # Init variables
        chi_thres = 0.103  # chi square 2DoF 95% confidence
        associd = list()
        Hk_list = list()
        Vk_list = list()
        Sk_list = list()
        Rk_list = list()


        # For each observed line
        print('\n-------- Associations --------')
        for i in range(0, lines.shape[0]):

            # The polar lines are already in the robot frame 
            z = funcs.get_polar_line(lines[i],[0,0,0])

            # Variables for finding minimum
            minD = 1e9
            minj = -1

            # For each line in the known map
            for j in range(0, self.map.shape[0]):

                # Compute matrices
                #The Jacobian needs the polar line of the map in the world frame
                h = funcs.get_polar_line(self.map[j],[0,0,0]) 
                H = self.jacobianH(h,self.xk)
                #Then for the innovation, the map line has to be in the robot frame
                h = funcs.get_polar_line(self.map[j],self.xk)
                v = z-h
                S = H @ self.Pk @ np.transpose(H) + self.Rk

                # Mahalanobis distance
                D = np.transpose(v)@ np.linalg.inv(S) @v

                # Optional: Check if observed line is longer than map
                ########################################################
                islonger = False

                # Check if the obseved line is the one with smallest
                # mahalanobis distance
                if np.sqrt(D) < minD and not islonger:
                     minj = j
                     minz = z
                     minh = h
                     minH = H
                     minv = v
                     minS = S
                     minD = D

            # Minimum distance below threshold
            if minD < chi_thres:
                 print("\t{} -> {}".format(minz, minh))
                 # Append results
                 associd.append([i, minj])
                 Hk_list.append(minH)
                 Vk_list.append(minv)
                 Sk_list.append(minS)
                 Rk_list.append(self.Rk)

        return Hk_list, Vk_list, Sk_list, Rk_list

    # ==========================================================================
    def update_position(self, Hk_list, Vk_list, Sk_list, Rk_list):
        """
        Update the position of the robot according to the given matrices.

        The matrices contain the current position and the data association
        parameters. All input lists have the same lenght.

        Input:
          Hk_list : list of 2x3 matrices (jacobian)
          Yk_list : list of 2x1 matrices (innovation)
          Sk_list : list of 2x2 matrices (innovation uncertainty)
          Rk_list : list of 2x2 matrices (measurement uncertainty)
        """
        # Compose list of matrices as single matrices
        n = len(Hk_list)
        H = np.zeros((2 * n, 3))
        v = np.zeros((2 * n))
        S = np.zeros((2 * n, 2 * n))
        R = np.zeros((2 * n, 2 * n))
        for i in range(n):
            H[2 * i:2 * i + 2, :] = Hk_list[i]
            v[2 * i:2 * i + 2] = Vk_list[i]
            S[2 * i:2 * i + 2, 2 * i:2 * i + 2] = Sk_list[i]
            R[2 * i:2 * i + 2, 2 * i:2 * i + 2] = Rk_list[i]

        # There is data to update
        if not n > 0:
            return

        #EKF update - Kalman Gain and update
        K = self.Pk @ np.transpose(H) @ np.linalg.inv(S)
        #Updating of the robot state
        self.xk += K @ v
        #updating of the robot uncertainty 
        self.Pk = (np.eye(3)-(K@H))@self.Pk

    # ==========================================================================
    def jacobianH(self, lineworld, xk):
        """
        Compute the jacobian of the get_polar_line function.

        It does it with respect to the robot state xk (done in pre-lab).
        """
  
        #Only the angle of the world lines interferes in the Jacobian matrix 
        psi_w   = lineworld[1]
        #Auxiliary variables defined to build the Jacobian matrix in a more readable way. 
        a1      = xk[0]**2 + xk[1]**2
        a2      = np.arctan2(xk[1],xk[0]) - psi_w
        #Partial derivatives of the measurement model 
        dh1_x    = -xk[1]*np.sin(a2)*(np.sqrt(a1)/a1) - (xk[0]/np.sqrt(a1))*np.cos(a2)
        dh1_y    =  xk[0]*np.sin(a2)*(np.sqrt(a1)/a1) - (xk[1]/np.sqrt(a1))*np.cos(a2)
        #Jacobian H
        H = np.array([[dh1_x,dh1_y,0],[0,0,-1]])

        return H
