import numpy as np
import math
import csv

class Odom:
    '''
    Odom Class to encapsulate information provided by an Odometer.
    It stores:
    - time : float representing the timestamp of the reading
    - u    : a numpy vector of dimension (2,) representing the measured linear and angular velocity
    - Q    : a numpy matrix of dimension (2,2) representing the covariance matrix of the measurement
    - dt   : a float representing the elapsed time since the last reading 
    '''
    def __init__(self, time, u, Q, dt):
        self.time = time
        self.u = u
        self.Q = Q
        self.dt = dt
# end class Odom


def ReadOdomCSV(csv_file):
    '''
    Read a CSV containing odometry information and return an ordered list of Odoms
    Input:
        - csv_file: string containing the path of the csv file to read
    Return:
        - list of Odom objects (ordered by time)
    '''
    data = []
    with open(csv_file, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        next(csv_reader) #Skip header
        for row in csv_reader:
            time = float(row[0])
            dt = float(row[1])
            u = np.array(row[2:4]).astype(float)
            Q = np.array([row[4:6],row[6:8]]).astype(float) + np.array([[0.001, 0], [0, 0.001]])
            data.append(Odom(time, u, Q, dt))

    return data



def ReadGroundTruthCSV(csv_file):
    '''
    Read a CSV containing the ground truth path andreturn an ordered list of np.arrays of shape (3,) 
    representing the x,y,\theta pose of the robot. 
    Input:
        - csv_file: string containing the path of the csv file to read
    Return:
        - list of  np.arrays of shape (3,) (ordered by time)
    '''
    gt_data = []
    gt_time = []
    with open(csv_file, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        next(csv_reader) #Skip header
        for row in csv_reader:
            time = float(row[0])
            gt_data.append(np.array(row[1:4]).astype(float))
            gt_time.append(time)
    return gt_data, gt_time