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
        self.B = np.eye(self.__sm.get_num_of_states())  # Identity matrix
        self.readings = []
        self.et = []  # Evidence, (List containing observations at time t)
        self.t = 1  # Time step t, initially 1

    # @function filter Returns a length 64 numpy array with probabilities for each position based on
    # forward backwards smoothing. Note that predictions always stay k-i moves behind!
    #
    # @param sensorR: Representing a sensor reading. (int)
    def filter(self, sensorR, lag=5):
        num_states = self.__sm.get_num_of_states()  # (int) 64
        pos = self.__sm.reading_to_position(sensorR)
        print('sensorR', sensorR)
        print('pos: ', pos)
        self.et.append(self.__om.get_o_reading(sensorR))  # Append the observation matrix for the current reading
        self.readings.append(sensorR)
        Ot = self.__om.get_o_reading(sensorR)  # Get diagonal matrix containing P(et|x)
        if self.t > lag:
            self.__f = self.__forward_filter(self.et)
            if len(self.et) > lag:
                self.et.pop(0)  # Remove oldest evidence
            self.B = self.__backward_filter(self.et)
        else:
            # Update backward transformation matrix B without using lagged evidence
            self.B = self.B @ np.linalg.inv(self.__tm.get_T_transp() + 1e-7 * np.eye(num_states)) @ Ot

        self.t += 1
        # Debug: Print statements for debugging
        print("Time Step:", self.t)
        print("Observation Matrix (Ot):", Ot)
        print("Forward Matrix (__f):", self.__f)
        print("Backward Matrix (B):", self.B)

        if self.t > lag:
            # Return normalized result of f × B[:, 1]
            result = self.__f @ self.B[:, 1]
            print("Result before normalization:", result)
            result = result / np.sum(result)
            print("Normalized Result:", result)
            print('Returning: ', result)
            return result
        else:
            # No result to return before the lag window is filled!
            print('Returning: ', self.__f)
            return self.__f
        # TODO: Uses the algorithm on last slide: file:///C:/Users/haral/Downloads/ProbReasTime2024.pdf

    def __forward_filter(self, et):
        num_states = self.__sm.get_num_of_states()  # (int) 64
        num_readings = self.__om.get_nr_of_readings()  # aka n (int)

        ### FORWARD FILTERING
        f = np.zeros((num_states, num_readings))  # Create a 2d array, Store forwards data
        T0 = np.ones(num_states) / num_states  # Uniform distribution
        T = self.__tm.get_T_transp()  # Transpose the matrix to align with formula inside Simons lecture notes
        f[:, 0] = et[0] @ T @ T0
        f[:, 0] /= np.sum(f[:, 0])

        # Debug: Print statements for forward filtering
        print("Forward Filtering - Initial Observation (t=0):", et[0])
        print("Forward Filtering - Initial Forward Matrix (t=0):", f[:, 0])

        for i in range(1, num_readings): # TODO: Leads to out of bounds
            f[:, i] = et[i] @ T @ f[:, i - 1]  # TODO: This can return none?
            f[:, i] /= np.sum(f[:, i])

            # Debug: Print statements for forward filtering
            print(f"\nForward Filtering - Observation (t={i}):", et[i])
            print(f"Forward Filtering - Forward Matrix (t={i}):", f[:, i])

        return f

    def __backward_filter(self, et):
        num_states = self.__sm.get_num_of_states()  # (int) 64
        num_readings = self.__om.get_nr_of_readings()  # aka n (int)
        TT = np.transpose(self.__tm.get_T_transp())

        # Initialize matrices for backward and smoothed estimates
        b = np.zeros((num_states, num_readings))
        s = np.zeros((num_states, num_readings))

        # Handle the last case separately
        b[:, -1] = np.array([[1], [1]])
        s[:, -1] = self.__f[:, -1] * b[:, -1]
        s[:, -1] /= np.sum(s[:, -1])

        # Debug: Print statements for backward filtering
        print("Backward Filtering - Initial Backward Matrix (t=T):", b[:, -1])
        print("Backward Filtering - Initial Smoothed Matrix (t=T):", s[:, -1])

        # Counting backwards
        for i in range(num_readings - 2, -1, -1):
            b[:, i] = TT @ et[i] @ b[:, i + 1]
            s[:, i] = self.__f[:, i] * b[:, i]
            s[:, i] /= np.sum(s[:, i])

            # Debug: Print statements for backward filtering
            print(f"\nBackward Filtering - Observation (t={i}):", et[i])
            print(f"Backward Filtering - Backward Matrix (t={i}):", b[:, i])
            print(f"Backward Filtering - Smoothed Matrix (t={i}):", s[:, i])

        return s
