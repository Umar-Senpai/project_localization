import numpy as np
import math
from utils import angle_wrap
import probabilistic_lib.functions as funcs
from PingerWithIDMeasurement import PingerWithIDMeasurement

class EKFSlam:
    '''
    Class to encapsulate the EKFSlam Problem.
    It stores:
        - x_: np.array representing the robot and map state x = [xr | xl1 | xl2 .... xln], where xr = [x,y,\theta] of the robot and xl = [x,y] of a landmark
        - P_: np matrix representing the robot and map covariance matrix
        - n_landmarks_ : int representing the number of landmarks in the map
        - landmark_size_ : int representing the variables that represent a landmark (e.g., for pingers its 2: [x,y])

    Note that all internal variables finish with an underscore. This is done in order to differenciate it from the class methods (and its usually standard to represent object variables)
    
    Since x_ and P_ will increase when new landmarks are detected, x_ and P_ matrices are pre-allocated for a maximum number of landmarks.
    Therefore, special care has to be taken in order to use or write to the right sections of x_ and P_ for operations, and not overwrite them with numpy arrays
    For this reason, a set of functions are built so we can access and write this data wisely
    A set of "getters" (functions that return information of the class) are defined to access data. They will return numpy views, which allow 
    to define subvectors and submatrices of x_ and P_ by reference (so if we modify them, it will modify the parent)
    A set of "setters" (functions to write information to the class) are defined to modify x_ and P_. The idea is to do it only here, so we don't do mistakes from outside the class
    '''
    def __init__(self, x_robot_init, P_robot_init, max_landmarks = 100, landmark_size = 2, map = None) -> None:
        max_x_size = 3 + max_landmarks * landmark_size
        self.x_ = np.zeros(max_x_size,)
        self.P_ = np.zeros((max_x_size, max_x_size))

        self.x_[0:3] = x_robot_init
        self.P_[0:3, 0:3] = P_robot_init

        self.Rk = np.array([[0.04, 0],
                            [0, 0.04]])

        self.n_landmarks_ = 0
        self.landmark_size_ = landmark_size
        if map is not None:
            for p in map:
                i = self.n_landmarks_
                self.x_[3 + i*2 : 5 + i*2] = p
                self.P_[3+i*2 : 5 + i*2, 3+i*2 : 5 + i*2] = np.eye(2) * 0.05
                self.n_landmarks_+=1

    #Getters
    ##Return views to self.x_ and self.P_ for efficiency and readibility!

    def x(self):
        '''
        return a numpy view of shape (3 + n * s,) where n is the number of landmarks and s is the landmark size
        '''
        return self.x_[0:3 + self.n_landmarks_ * self.landmark_size_]
    
    def P(self):
        '''
        return a numpy view of shape (3 + n*s, 3 + n*s) representing the robot state covariance
        '''
        return self.P_[0:3 + self.n_landmarks_ * self.landmark_size_, 0:3 + self.n_landmarks_ * self.landmark_size_]

    def xr(self):
        '''
        return a numpy view of shape (3,) representing the robot state [x,y,\theta]
        '''
        return self.x_[0:3]

    def Prr(self):
        '''
        return a numpy view of shape (3,3) representing the robot state covariance
        '''
        return self.P_[0:3, 0:3]

    def xl(self, idx):
        '''
        return a numpy view of shape (s,) representing a landmark state (s is 2 for a pinger [x,y])
        Input:
            - idx: an integer representing the landmark position in the map
        '''
        if idx >= self.n_landmarks_:
            return None
        f = 3 + idx * self.landmark_size_
        t = 3 + (idx + 1) * self.landmark_size_
        return self.x_[f:t]

    def Prm(self):
        '''
        return a numpy view of shape (3, n * s) representing the robot-map correlation matrix where n is the number of landmarks and s is the landmark size
        '''
        #
        return self.P_[0:3,3:3 + self.n_landmarks_*self.landmark_size_]

    def Pmr(self):
        '''
        return a numpy view of shape (n * s, 3) representing the map-robot correlation matrix where n is the number of landmarks and s is the landmark size
        '''
        #
        return self.P_[3:3 + self.n_landmarks_*self.landmark_size_,0:3]

    def Pmm(self):
        '''
        return a numpy view of shape (n * s, n * s) representing the map covariance matrix where n is the number of landmarks and s is the landmark size
        '''
        #
        return self.P_[3:3 + self.n_landmarks_*self.landmark_size_,3:3 + self.n_landmarks_*self.landmark_size_]

    def Prl(self, idx):
        '''
        return a numpy view of shape (3, s) representing a robot-landmark correlation matrix where s is the landmark size
        Input:
            - idx: an integer representing the landmark position in the map
        '''
        
        return self.P_[0:3, 3 + idx * self.landmark_size_: 5 + idx * self.landmark_size_]

    def Plr(self, idx):
        '''
        return a numpy view of shape (s, 3) representing a landmark-robot correlation matrix where s is the landmark size
        Input:
            - idx: an integer representing the landmark position in the map
        '''
        #
        return self.P_[3 + idx * self.landmark_size_: 5 + idx * self.landmark_size_, 0:3]

    def Plm(self, idx):
        '''
        return a numpy view of shape (s, n * s) representing a landmark-map correlation matrix where n is the number of landmarks and s is the landmark size
        Input:
            - idx: an integer representing the landmark position in the map
        '''
        #
        return self.P_[3 + idx * self.landmark_size_: 5 + idx * self.landmark_size_,3:3 + self.n_landmarks_*self.landmark_size_]

    def Pml(self, idx):
        '''
        return a numpy view of shape (n * s, s) representing a map-landmark correlation matrix where n is the number of landmarks and s is the landmark size
        Input:
            - idx: an integer representing the landmark position in the map
        '''
        #
        return self.P_[3:3 + self.n_landmarks_*self.landmark_size_,3 + idx * self.landmark_size_: 5 + idx * self.landmark_size_]


    def Pll(self, idx):
        '''
        return a numpy view of shape (s,s) representing the landmark covariance matrix where s is the size of the landmark
        Input:
            -idx : an integer representing the landmark position in the map
        '''
        #
        return self.P_[3 + idx * self.landmark_size_: 5 + idx * self.landmark_size_, 3 + idx * self.landmark_size_: 5 + idx * self.landmark_size_]

    # THE EKF Methods, assuming the map is in the state!
    def prediction(self, u, Q, dt): 
        '''
        Compute the prediction step of the EKF.
        Input: 
            - u : np array representing the control signal
            - Q : np matrix representing the noise of the control signal
            - dt: float representing the increment of time
        Return:
            - xr: np array of shape (3,) representing the predicted robot state
            - Prr: np matrix of shape (3,3) representing the predicted robot covariance
            - Prm: np matrix of shape (3, n * s) representing the predicted robot-map correlation matrix
        '''
        assert isinstance(u, np.ndarray)
        assert isinstance(Q, np.ndarray)
        assert isinstance(dt, float)
        
        xr = self.calculate_f(u, dt)
        A = self.calculate_Jfx(u, dt)
        W = self.calculate_Jfw(dt)
        Prr = A @ self.Prr() @ np.transpose(A) + W @ Q @ np.transpose(W)
        Prm = A @ self.Prm()
        
        return [xr, Prr, Prm] # We don't need Pmr since Prm = Prm.T

    
    def update(self, z, R, lid, hx, Hr, Hl, Hv):
        '''
        Compute the update step of the EKF.
        Input: 
            - z : np array representing the measurement
            - R : np matrix representing the noise of the measurement
            - lid: int representing the landmark id of the measurement
            - hx: np array representing the expected measurement
            - Hr: np matrix representing the jacobian of hx with respect to xr
            - Hl: np matrix representing the jacobian of hx with respect to xl
            - Hv: np matrix representing the jacobian of hx with respect to v
        Return:
            - x: np array of shape (3 + n * s,) representing the updated robot and map state
            - P: np matrix of shape (3 + n*s, 3 + n*s) representing the updated robot and map covariance
        '''
        assert isinstance(z, np.ndarray)
        assert isinstance(R, np.ndarray)
        assert isinstance(hx, np.ndarray)
        assert isinstance(Hr, np.ndarray)
        assert isinstance(Hl, np.ndarray)
        assert isinstance(Hv, np.ndarray)
        assert isinstance(lid, int)
 
        # Compute innovation
        y = z - hx
        for i in range(1, len(y), 2): # Assuming all measurements are [distance, angle]
            y[i] = angle_wrap(y[i])
        
        H = np.concatenate((Hr, Hl), axis=1)
        P_E_row1 = np.concatenate((self.Prr(), self.Prl(lid)), axis=1) 
        P_E_row2 = np.concatenate((self.Plr(lid), self.Pll(lid)), axis=1)
        P_E =  np.concatenate((P_E_row1, P_E_row2), axis=0)
        E = H @ P_E @ H.transpose()
        Z = E + R
        P_K_row2 = np.concatenate((self.Pmr(), self.Pml(lid)), axis=1)
        P_K = np.concatenate((P_E_row1, P_K_row2), axis=0) 
        K = P_K @ H.transpose() @ np.linalg.inv(Z)
        x = self.x_[0:3 + self.n_landmarks_ * self.landmark_size_] + K @ y
        P = self.P_[0:3 + self.n_landmarks_ * self.landmark_size_, 0:3 + self.n_landmarks_ * self.landmark_size_] - K @ Z @ K.transpose()
        
        return [x, P]

    # Setters:  Functions to apply changes!
    def applyPrediction(self, xr, Prr, Prm):
        '''
        Apply a prediction (Be careful fill self.x_ and self.P_, but don't overwrite it!)
        Input: 
            - xr: np array of shape (3,) representing the predicted robot state
            - Prr: np matrix of shape (3,3) representing the predicted robot covariance
            - Prm: np matrix of shape (3, n * s) representing the predicted robot-map correlation matrix
        '''
        self.x_[0:3] = xr
        self.P_[0:3, 0:3] = Prr
        self.P_[0:3, 3: 3 + self.n_landmarks_*self.landmark_size_] = Prm
        self.P_[3: 3 + self.n_landmarks_*self.landmark_size_, 0:3] = Prm.transpose()

    def applyUpdate(self, x, P):
        '''
        Apply a update (Be careful fill self.x_ and self.P_, but don't overwrite it!)
        Input: 
            - x: np array of shape (3 + n*s,) representing the updated state
            - P: np matrix of shape (3 + n*s, 3 + n*s) representing the updated state covariance
        '''
        #print('______', self.n_landmarks_)
        self.x_[0:3 + self.n_landmarks_ * self.landmark_size_] = x
        self.P_[0:3 + self.n_landmarks_ * self.landmark_size_, 0:3 + self.n_landmarks_ * self.landmark_size_] = P

    def add_landmark(self, xl, Gr, Gz, R):
        '''
        Add a new landmark to the state!
        Input: 
            - xl: np array representing the landmark state 
            - Gr: np matrix representing the jacobian of g(xr, z) with respect to xr
            - Gz: np matrix representing the jacobian of g(xr, z) with respect to z 
        Return:
            - idx: int representing the landmark position in the map
        '''
        idx = self.n_landmarks_
        #
        #

        ##Add landmark to the state vector (xl)
        self.x_[3 + idx*2 : 5 + idx*2] = xl
        ##Add landmark uncertainty (Pll)
        self.P_[3 + idx*2 : 5 + idx*2 , 3 + idx*2 : 5 + idx*2] = Gr @ self.Prr() @ Gr.transpose() + Gz @ R @ Gz.transpose()
        ## Fill the cross-variance of the new landmark with the rest of the state (Plx)
        self.P_[3 + idx*2 : 5 + idx*2 , 0:3 + idx*2] = Gr @ np.concatenate((self.Prr(), self.Prm()), axis=1)
        ## Fill the cross-variance of the rest of the state with the new landmark (Pxl)
        self.P_[0:3 + idx*2, 3 + idx*2 : 5 + idx*2] = (self.P_[3 + idx*2 : 5 + idx*2 , 0:3 + idx*2]).transpose()
        
        self.n_landmarks_ += 1 # Update the counter of landmarks
        return idx # Return the position of the landmark in our state


    # The Differential Drive Prediction equations (From last lab)
    def calculate_f(self, u, dt) -> np.ndarray:
        # x = self.x_[0] + math.cos(self.x_[2]) * u[0] * dt
        # y = self.x_[1] + math.sin(self.x_[2]) * u[0] * dt
        # theta = angle_wrap(self.x_[2] + u[1] * dt)
        # return np.array([x,y,theta])
        return funcs.comp(self.x_, u)

    def calculate_Jfx(self, u, dt) -> np.ndarray:
        # return np.array([[1, 0, -(u[0] * dt)*math.sin(self.x_[2])], 
        #                  [0, 1,  (u[0] * dt)*math.cos(self.x_[2])], 
        #                  [0, 0,   1]])

        return np.array([[1, 0, -np.sin(self.x_[2]) * u[0] - np.cos(self.x_[2]) * u[1]],
                        [0, 1, np.cos(self.x_[2]) * u[0] - np.sin(self.x_[2]) * u[1]],
                        [0, 0, 1]])
        
    def calculate_Jfw(self, dt = None) -> np.ndarray:
        # return np.array([[math.cos(self.x_[2]) * dt, 0], 
        #                  [math.sin(self.x_[2]) * dt, 0], 
        #                  [0                        , dt]])

        return np.array([[np.cos(self.x_[2]), -np.sin(self.x_[2]), 0],
                        [np.sin(self.x_[2]), np.cos(self.x_[2]), 0],
                        [0, 0, 1]])

    def is_data_associated(self, corners):
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
        # chi_thres = 0.103  # chi square 2DoF 95% confidence
        chi_thres = 0.103
        associd = dict()
        distance = dict()
        Hk_list = list()
        Vk_list = list()
        Sk_list = list()
        Rk_list = list()


        # For each observed line
        #print('\n-------- Associations --------')
        for i in range(0, corners.shape[0]):

            # The polar lines are already in the robot frame
            z = PingerWithIDMeasurement.h([0,0,0], corners[i]) #Converting Corners in distance and angle 
            # z = funcs.get_polar_line(corners[i],[0,0,0])

            # Variables for finding minimum
            minD = 1e9
            minj = -1

            # For each line in the known map
            for j in range(0, self.n_landmarks_):

                # Compute matrices
                #The Jacobian needs the polar line of the map in the world frame
                # h = funcs.get_polar_line(self.map[j],[0,0,0])
                h = PingerWithIDMeasurement.h(self.xr(), self.xl(j)) 
                # H = self.jacobianH(h,self.xk)
                H = PingerWithIDMeasurement.Jhxr(self.xr(), self.xl(j))
                #Then for the innovation, the map line has to be in the robot frame
                # h = funcs.get_polar_line(self.map[j],self.xk)
                v = z - h
                S = H @ self.Prr() @ np.transpose(H) + PingerWithIDMeasurement.Jhv() @ self.Rk @ np.transpose(PingerWithIDMeasurement.Jhv())

                # Mahalanobis distance
                D = np.transpose(v)@ np.linalg.inv(S) @v

                # Optional: Check if observed corner is longer than map
                ########################################################
                islonger = False

                # Check if the obseved corner is the one with smallest
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
            #print('---- minD', minD)
            if minD < chi_thres:
                 #print("\t{} -> {}".format(minz, minh))
                 # Append results
                 associd[i] = minj
                 Hk_list.append(minH)
                 Vk_list.append(minv)
                 Sk_list.append(minS)
                 Rk_list.append(self.Rk)
            distance[i] = minD

        return associd, distance