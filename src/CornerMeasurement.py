import numpy as np
import math
import csv
from utils import angle_wrap


class CornerMeasurement:
    '''
    Class to encapsulate information provided by a range&bearing sensor that can detect corners located in the world.
    The sensor is assumed to be located at the center of the robot.
    It stores:
    - time : float representing the timestamp of the reading
    - z    : a numpy vector of dimension (2,) representing the measured distance and angle
    - R    : a numpy matrix of dimension (2,2) representing the covariance matrix of the measurement
    - id   : int representing a unique identifier of the corner (it solves the association problem)
    '''
    def __init__(self, time, z, R, id):
        self.time = time
        self.z = z
        self.R = R
        self.id = id

    def h(xr, xl):
        '''
        Compute the expected measurement z = h(xr, xl, v), where xr is the robot state [x,y,\theta] and xl the corner state [x,y]
        Input:
        - xr: numpy array of shape (3,) representing the robot state [x,y,\theta]
        - xl: numpy array of shape (2,) representing the [x,y] position of the detected corner in the world
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
        - xl: numpy array of shape (2,) representing the [x,y] position of the detected corner in the world
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
        - xl: numpy array of shape (2,) representing the [x,y] position of the detected corner in the world
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
        - xl: numpy array of shape (2,) representing the [x,y] position of the detected corner in the world
        return: numpy matrix of shape (2, 2) (The Jacobian)
        '''
        # Noise is assumed independent (z = h(x) + v)
        return np.eye(2)

    def g(xr, z):
        '''
        Compute the inverse observation xl = g(xr, z, v), where xl the corner state [x,y], xr is the robot state [x,y,\theta] 
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
        Compute the Jacobian of xl = g(xr, z, v), where xl the corner state [x,y], xr is the robot state [x,y,\theta] 
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
        Compute the Jacobian of xl = g(xr, z, v), where xl the corner state [x,y], xr is the robot state [x,y,\theta] 
        and z is the measure [dist, angle] (contained in self.z) with respect to z at point (xr, z)
        Input:
        - xr: numpy.array of shape (3,) representing the robot state [x,y,\theta]
        return: numpy matrix of shape (2, 3) (The Jacobian)
        '''
        # z = self.z
        d_d = [np.cos(xr[2] + z[1]), -1 * z[0] * np.sin(xr[2] + z[1])]
        d_alpha = [np.sin(xr[2] + z[1]),  z[0] * np.cos(xr[2] + z[1])]
        return np.array([d_d, d_alpha])