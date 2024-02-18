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
        self.observations = []  # (List containing sensorR at time t)

    # @function filter: Returns "normal" forward filtering
    #
    # @param sensorR: Representing a sensor reading. (int)
    def filter(self, sensorR):
        self.__f = self.__om.get_o_reading(sensorR) @ self.__tm.get_T_transp() @ self.__f
        self.__f /= np.sum(self.__f)
        return self.__f

    # @function smoothing: Performs FB smoothing
    #
    # @param sensorR: Representing a sensor reading. (int)
    # @param lag: Amount of lag steps. (int)
    def smoothing(self, sensorR, lag=5):
        self.observations.append(sensorR)
        if len(self.observations) <= lag:
            return self.__f
        Ok = self.observations[0]
        self.observations = self.observations[1:]
        self.__f = self.filter(Ok)  # <----- CHANGED AFTER PEER REVIEW (prev. duplicate code)
        b = np.ones_like(self.__f)
        for observation in reversed(self.observations):
            b = self.__tm.get_T() @ self.__om.get_o_reading(observation) @ b  # <----- CHANGED AFTER PEER REVIEW (prev. self.__tm.get_T_transp())
        s = self.__f * b
        return s / np.sum(s)
