from EKFSlamBase import EKFSlam
from Odometry import Odom, ReadOdomCSV, ReadGroundTruthCSV
from PingerWithIDMeasurement import PingerWithIDMeasurement, ReadPingsCSV
from utils import angle_wrap

# Matplotlib imports
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib.patches as patches
from matplotlib import transforms as trans
import matplotlib.collections as mc

import numpy as np
import math
import datetime

class SLAMAnimation(EKFSlam):
    '''
    Class to encapsulate the EKF SLAM Animation using range&bearing measurements from pingers!
    It inherits from EKFSLam so it has all its methods (x(), P() ...) and variables (x_, P_, n_landmarks_ landmark_size_)
    Important!!!! We will add landmarks to the map in the order that we measure them. So the position of a landmark with id = 2, will not be necessarily at position 2 in the map.
    Therefore, it is important to keep track of the positions of the landmarks based on the id. This is the purpose of the dictionary self.pings_id_to_idx.

    It stores (apart from the inherited variables):
        - pings_id_to_idx: a dictionary to relate pingers ids to positions in our internal map (the state). 
            Example: Assume a measurement of a pinger with id = 145. It was the first pinger ever measured, so it was put the first one in our map. Then, self.pings_id_to_idx[145] = 0 (Now you can get, for example, the pose of the landmark using self.xl(idx))
        - odom_data, pings_data : the lists of odometry and measurements parsed from csvs
        - The rest is for plotting purposes
    '''

    def __init__(self, xr_init, Pr_init, odom_data, gt_path, pings_data, initial_map = None) -> None:
        '''
        Input:
            - xr_init: np array of shape (3,) representing the initial robot state
            - Pr_init: np array of shape (3,3) representing the initial robot covariance
            - odom_data : a list of Odoms ordered by time
            - pings_data: a list of PingsWithIDMeasurements orered by time (same size as odom_data)
            - gt_path : a list of ground truth positions of the robot (known because its synthetic data)
            - initial_map: a list of landmark poses in case you want to initialize the state with map (useful for testing purposes)
        '''
        super().__init__(xr_init, Pr_init, 100, 2, initial_map)
        
        self.odom_data = odom_data
        self.pings_data = pings_data
        self.gt_path = gt_path

        self.pings_id_to_idx = {}
        if initial_map is not None:
            for i in range(len(initial_map)):
                self.pings_id_to_idx[i] = i

        self.traj_x = []
        self.traj_y = []
           
        self.fig = plt.figure() 
        self.ax = self.fig.add_subplot(111, autoscale_on=False, xlim=(-20, 60), ylim=(-15, 65))
        self.ax.set_title('EKFSlam using range&bearing pingers')
        self.ax.set_aspect('equal')
        self.ax.set(xlabel='x[m]', ylabel='y[m]')
        self.ax.grid()
        self.time = datetime.datetime.now()

    # Filter iteration
    def iterate(self, time_idx):
        ## EKF
        # prediction
        start_time = datetime.datetime.now()

        odom = self.odom_data[time_idx]
        [xr, Pr, Prm] = self.prediction(odom.u, odom.Q, odom.dt)
        self.applyPrediction(xr, Pr, Prm)
        
        after_prediction_time = datetime.datetime.now()

        #update & initialization
        pings = self.pings_data[time_idx]   
        for p in pings:
            assert isinstance(p, PingerWithIDMeasurement)
            if p.id in self.pings_id_to_idx.keys(): #Known landmark! Do Upate
                #print("Update ", p.id)
                idx = self.pings_id_to_idx[p.id]
                xr = self.xr()
                xl = self.xl(idx)
                [x, P] = self.update(p.z,
                                    p.R,
                                    idx,
                                    PingerWithIDMeasurement.h(xr, xl), 
                                    PingerWithIDMeasurement.Jhxr(xr, xl),
                                    PingerWithIDMeasurement.Jhxl(xr, xl),
                                    PingerWithIDMeasurement.Jhv())
                self.applyUpdate(x, P)   
            else: #Unknown landmark! Do initialization
                #print("Initialize ", p.id)
                idx = self.add_landmark(p.g(self.xr()), 
                                        p.Jgxr(self.xr()), 
                                        p.Jgz(self.xr()), 
                                        p.R)
                self.pings_id_to_idx[p.id] = idx

        after_update_time = datetime.datetime.now()
                                               
        #Do animation!
        return_plot = self.plot(time_idx)
        
        after_plot_time = datetime.datetime.now()
        print("iteration ", time_idx)
        print("Plot time (+ anim interval): ", (start_time - self.time).microseconds)
        print("Prediction time: ",(after_prediction_time - start_time).microseconds )
        print("Update time: ",(after_update_time - after_prediction_time).microseconds )
        print("Prepare Plot time: ",(after_plot_time - after_update_time).microseconds )
        print("------------------")
        self.time = after_plot_time
        return return_plot

        
    ## ANIMATION METHODS!!
    def init_animation(self):
               # Drawing preparation (figure with one subplot)
       
        self.anim_path, = self.ax.plot([], [], 'r-', lw=1) # Line for displaying the path
        self.anim_robot = patches.Rectangle((-0.1,-0.05), 0.6, 0.3, angle=0, color='r')
        self.ax.add_patch(self.anim_robot)

        gt_x = [x[0] for x in self.gt_path]
        gt_y = [x[1] for x in self.gt_path]
        #self.anim_gt_path, = self.ax.plot(gt_x, gt_y, 'b-', lw=1) # Line for displaying the path

        self.anim_gt_robot = patches.Rectangle((-0.1,-0.05), 0.6, 0.3, angle=0, color='b')
        self.anim_robot_cov = patches.Ellipse((0.0, 0.0), 0.9, 0.5, 0, edgecolor='r', lw=1, facecolor='none')
        self.ax.add_patch(self.anim_robot_cov)
        self.ax.add_patch(self.anim_gt_robot)
        
        self.detections_collection = mc.LineCollection([[(0,2),(0,1)]], linestyle='dashed')
        self.ax.add_collection(self.detections_collection)

        self.map_patches = []

        self.anim_landmarks_pos = []
        self.anim_landmarks_ellipse = []

        return self.anim_gt_robot, self.anim_path, self.anim_robot, self.anim_robot_cov, self.detections_collection

    def plot(self, idx):
         ## PLOT
        # Update trajectory path
        self.traj_x.append(self.x_[0])
        self.traj_y.append(self.x_[1])

        # Update the drawing...
        # Update robot believe mean
        #self.anim_path.set_data(self.traj_x, self.traj_y)
        self.anim_robot.set_transform(trans.Affine2D().rotate(self.x_[2]) + trans.Affine2D().translate(self.x_[0],self.x_[1])+  self.ax.transData )
        
        #Update robot believe ellipse (Uncertainty)
        v, w = np.linalg.eigh(self.P_[0:2,0:2])
        v = 2.0 * np.sqrt(5.991) * np.sqrt(v) # 5.991 is the chi-square value for 95% confidence in 2DOF
        u = w[0] / np.linalg.norm(w[0])
        ell_angle = np.arctan2(u[1] , u[0])
        self.anim_robot_cov.width = v[0]
        self.anim_robot_cov.height = v[1]
        self.anim_robot_cov.set_transform( trans.Affine2D().rotate(ell_angle) + trans.Affine2D().translate(self.x_[0],self.x_[1])+  self.ax.transData )
        
        # Update robot ground truth position
        self.anim_gt_robot.set_transform(trans.Affine2D().rotate(self.gt_path[idx][2]) + trans.Affine2D().translate(self.gt_path[idx][0],self.gt_path[idx][1])+  self.ax.transData )
     
        for i in range(self.n_landmarks_):
            pos = self.xl(i)
            #Update robot believe ellipse (Uncertainty)
            v, w = np.linalg.eigh(self.Pll(i))
            v = 2.0 * np.sqrt(5.991) * np.sqrt(v) # 5.991 is the chi-square value for 95% confidence in 2DOF
            u = w[0] / np.linalg.norm(w[0])
            ell_angle = np.arctan2(u[1] , u[0])
            if i >= len(self.map_patches):
                self.map_patches.append(patches.Ellipse((0.0, 0.0), 10000, 10000, 0, edgecolor='b', lw=1, facecolor='none'))  
                self.ax.add_patch(self.map_patches[i])

            self.map_patches[i].width = v[0]
            self.map_patches[i].height = v[1]
            self.map_patches[i].set_transform( trans.Affine2D().rotate(ell_angle) + trans.Affine2D().translate(pos[0], pos[1])+  self.ax.transData )
           
        #Print Measurements
        pings = self.pings_data[idx] 
        if len(pings) > 0:
            lines = []
            for p in pings:
                assert isinstance(p, PingerWithIDMeasurement)
                xl = self.x_[0] + math.cos(self.x_[2] + p.z[1]) * p.z[0]
                yl = self.x_[1] + math.sin(self.x_[2] + p.z[1]) * p.z[0]
                lines.append([[self.x_[0], self.x_[1]], [xl, yl]])
   
            self.detections_collection.set_segments(lines)
        

        return (self.anim_gt_robot, self.anim_robot_cov, self.anim_robot, self.detections_collection) + tuple(self.map_patches)

    def plotGroundTruthMap(self, map):
        for p in map:
            pinger = plt.Circle((p[0],p[1]), 0.4)
            plt.gca().add_patch(pinger)



if __name__ == "__main__":
    x_init = np.array([0.0,0.0,0.0])
    P_init = np.eye(3)*0.0001

    odom_data = ReadOdomCSV("SLAMDataSet/odom.csv")
    [gt_data,gt_time] = ReadGroundTruthCSV("SLAMDataSet/ground_truth.csv")
    [pings_data, pingers_map] = ReadPingsCSV("SLAMDataSet/pings.csv")

    robot = SLAMAnimation(x_init, P_init, odom_data, gt_data, pings_data, None)
    # Use the following initialization if you want to start with the map known a priory (EKF only)
    # robot = SLAMAnimation(x_init, P_init, odom_data, gt_data, pings_data, pingers_map)
    # also, a no_pings.csv file can be used to execute dead reackoning

    robot.plotGroundTruthMap(pingers_map)
 
    pings_anim = anim.FuncAnimation(robot.fig, robot.iterate, range(0,len(odom_data)), init_func=robot.init_animation, interval=1 , blit=True, repeat=False)
    plt.show()