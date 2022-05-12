import numpy as np
import math
import csv
from utils import angle_wrap


class PingerWithIDMeasurement:
    '''
    Class to encapsulate information provided by a range&bearing sensor that can detect pingers located in the world.
    The sensor is assumed to be located at the center of the robot.
    It stores:
    - time : float representing the timestamp of the reading
    - z    : a numpy vector of dimension (2,) representing the measured distance and angle
    - R    : a numpy matrix of dimension (2,2) representing the covariance matrix of the measurement
    - id   : int representing a unique identifier of the pinger (it solves the association problem)
    '''
    def __init__(self, time, z, R, id):
        self.time = time
        self.z = z
        self.R = R
        self.id = id

    #def expected_measurement(self, x, mfeat):
    def h(xr, xl):
        '''
        Compute the expected measurement z = h(xr, xl, v), where xr is the robot state [x,y,\theta] and xl the pinger state [x,y]
        Input:
        - xr: numpy array of shape (3,) representing the robot state [x,y,\theta]
        - xl: numpy array of shape (2,) representing the [x,y] position of the detected pinger in the world
        Return: numpy array of shape (2,) representing the expected measurement [dist, angle]
        '''
        expected_distance = np.sqrt((xr[0]-xl[0])**2 + (xr[1]-xl[1])**2)
        expected_angle = angle_wrap(np.arctan2(xl[1]-xr[1], xl[0]-xr[0]) - xr[2])
        return np.array([expected_distance, expected_angle])


    def Jhxr(xr, xl) -> np.ndarray:
        '''
        Compute the Jacobian of h(xr, xl ,v) with respect to xr, at point (xr, xl)
        Input:
        - xr: numpy.array of shape (3,) representing the robot state [x,y,\theta]
        - xl: numpy array of shape (2,) representing the [x,y] position of the detected pinger in the world
        return: numpy matrix of shape (2, 3) (The Jacobian)
        '''
        d_x = (xr[0] - xl[0])/(np.sqrt((xr[0]-xl[0])**2 + (xr[1]-xl[1])**2))
        d_y = (xr[1] - xl[1])/(np.sqrt((xr[0]-xl[0])**2 + (xr[1]-xl[1])**2))
        d_theta = 0
        theta_x = (xl[1] - xr[1])/((xl[0] - xr[0])**2 + (xl[1] - xr[1])**2)
        theta_y = (xr[0] - xl[0])/((xl[0] - xr[0])**2 + (xl[1] - xr[1])**2)
        theta_theta = -1
        return np.array([[d_x, d_y, d_theta], [theta_x, theta_y, theta_theta]])

    def Jhxl(xr, xl) -> np.ndarray:
        '''
        Compute the Jacobian of h(xr, xl ,v) with respect to xl, at point (xr, xl)
        Input:
        - xr: numpy.array of shape (3,) representing the robot state [x,y,\theta]
        - xl: numpy array of shape (2,) representing the [x,y] position of the detected pinger in the world
        return: numpy matrix of shape (2, 2) (The Jacobian)
        '''
        d_x = (xl[0] - xr[0])/(np.sqrt((xr[0]-xl[0])**2 + (xr[1]-xl[1])**2))
        d_y = (xl[1] - xr[1])/(np.sqrt((xr[0]-xl[0])**2 + (xr[1]-xl[1])**2))
        theta_x = (xr[1] - xl[1])/((xl[0] - xr[0])**2 + (xl[1] - xr[1])**2)
        theta_y = (xl[0] - xr[0])/((xl[0] - xr[0])**2 + (xl[1] - xr[1])**2)
        return np.array([[d_x, d_y], [theta_x, theta_y]])
    
    def Jhv(xr = None, xl = None) -> np.ndarray:
        '''
        Compute the Jacobian of h(xr, xl ,v) with respect to v, at point (xr, xl)
        Input:
        - xr: numpy.array of shape (3,) representing the robot state [x,y,\theta]
        - xl: numpy array of shape (2,) representing the [x,y] position of the detected pinger in the world
        return: numpy matrix of shape (2, 2) (The Jacobian)
        '''
        # Noise is assumed independent (z = h(x) + v)
        return np.eye(2)

    def g(xr, z):
        '''
        Compute the inverse observation xl = g(xr, z, v), where xl the pinger state [x,y], xr is the robot state [x,y,\theta] 
        and z is the measure [dist, angle] (contained in self.z) 
        Input:
        - xr: numpy array of shape (3,) representing the robot state [x,y,\theta]
        Return: numpy array of shape (2,) representing the landmark expected position [x, y]
        '''
        # z = self.z
        x = xr[0] + z[0] * np.cos(z[1] + xr[2])
        y = xr[1] + z[0] * np.sin(z[1] + xr[2])
        return np.array([x, y])

    def Jgxr(xr, z) -> np.ndarray:
        '''
        Compute the Jacobian of xl = g(xr, z, v), where xl the pinger state [x,y], xr is the robot state [x,y,\theta] 
        and z is the measure [dist, angle] (contained in self.z) with respect to xr at point (xr, z)
        Input:
        - xr: numpy.array of shape (3,) representing the robot state [x,y,\theta]
        return: numpy matrix of shape (2, 3) (The Jacobian)
        '''
        # z = self.z
        dx = [1, 0, -1 * z[0] * np.sin(xr[2] + z[1])]
        dy = [0, 1,  z[0] * np.cos(xr[2] + z[1])]
        return np.array([dx, dy])

    def Jgz(xr, z) -> np.ndarray:
        '''
        Compute the Jacobian of xl = g(xr, z, v), where xl the pinger state [x,y], xr is the robot state [x,y,\theta] 
        and z is the measure [dist, angle] (contained in self.z) with respect to z at point (xr, z)
        Input:
        - xr: numpy.array of shape (3,) representing the robot state [x,y,\theta]
        return: numpy matrix of shape (2, 3) (The Jacobian)
        '''
        # z = self.z
        d_d = [np.cos(xr[2] + z[1]), -1 * z[0] * np.sin(xr[2] + z[1])]
        d_alpha = [np.sin(xr[2] + z[1]),  z[0] * np.cos(xr[2] + z[1])]
        return np.array([d_d, d_alpha])
#end class PingerWithIDMeasurement

   
def ReadPingsCSV(csv_file):
    '''
    Read a CSV containing Range and Bearing measurements information and return an ordered list of PingerWithIDMeasurement
    Input:
        - csv_file: string containing the path of the csv file to read
    Return:
        - data: list of lists of PingerWithIDMesurement objects, since in a specific time there can be multiple pings measured:
            Example: data[0] returns a list of three measurements [a,b,c]. data[1] returns an empty list [] (no measurements at time 1)
        - pingers_map: a list of np.array of shape (2,) representing the ground truth position of the pingers in the world
            You are not supposed to know if when doing SLAM, but is usefull for plotting purposes
    '''
    data = []
    pingers_map = []
    with open(csv_file, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        next(csv_reader) #Skip header
        row = next(csv_reader) #Map
        num_pingers = int(row[0])
        for i in range(num_pingers):
            pingers_map.append(np.array([row[1 + 2 * i],row[2 + 2*i]]).astype(float))
        for row in csv_reader:
            time = float(row[0])
            num_pings = int(row[1])
            pings = []
            step = 7
            for i in range(num_pings):
                pid = int(row[2 + i*step])
                z = np.array(row[3 + i * step: 5 + i * step ]).astype(float)
                R = np.array(row[5 + i * step: 9 + i * step ]).astype(float).reshape(2,2) + np.array([[0.1, 0], [0, 0.1]])
                pings.append(PingerWithIDMeasurement(time, z, R, pid))
            data.append(pings)
           
    return data, pingers_map