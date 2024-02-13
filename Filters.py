import random
import numpy as np

from models import *


#
# Add your Filtering / Smoothing approach(es) here
#
class HMMFilter:
    def __init__(self, probs, tm, om, sm):
        self.__tm = tm  # Transition model.
        self.__om = om  # Observation model.
        self.__sm = sm  # State model.
        self.__f = probs  # Initial probability distribution.

    # @function filter Returns a length 64 numpy array with probabilities for each position based on
    # forward backwards smoothing. Note that predictions always stay k-i moves behind!
    #
    # @param sensorR: Representing a sensor reading. (int)
    def filter(self, sensorR):

        ### FORWARD FILTERING

        num_states = self.__sm.get_num_of_states()  # (int) 64
        num_readings = self.__om.get_nr_of_readings()  # aka n (int)
        f = np.zeros((num_states, num_readings))  # Create a 2d array, Store forwards data
        T0 = np.ones(num_states) / num_states  # Uniform distribution
        T = self.__tm.get_T_transp()  # Transpose the matrix to align with formula inside Simons lecture notes
        O0 = self.__om.get_o_reading(0)
        f[:, 0] = O0 @ T @ T0
        f[:, 0] /= np.sum(f[:, 0])
        for i in range(1, num_readings):
            f[:, i] = self.__om.get_o_reading(sensorR) @ T @ f[:, i - 1]
            f[:, i] /= np.sum(f[:, i])

            ### BACKWARD FILTERING

        TT = np.transpose(T)
        b = np.zeros((num_states, num_readings))  # Store backwards data
        s = np.zeros((num_states, num_readings))  # Store smoothed estimate here
        b[:, num_readings - 1] = np.ones(num_states)  # Handle first case. This is a "one" vector by default
        s[:, num_readings - 1] = f[:, num_readings - 1] * b[:, num_readings - 1]  # Perform element-wise multiplication
        s[:, num_readings - 1] /= np.sum(s[:, num_readings - 1])  # To deal w/ alpha

        for i in range(num_readings - 2, -1, -1):
            b[:, i] = TT @ self.__om.get_o_reading(sensorR) @ b[:, i + 1]
            s[:, i] = f[:, i] * b[:, i]
            s[:, i] /= np.sum(s[:, i])

        s = np.prod(s, axis=1)  # Convert the (64,17) Matrix to a (64,) array

        return s
