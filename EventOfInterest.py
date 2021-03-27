"""
 This file is part of BeSafeBox Android application.
 Copyright (C) 2019  Tomáš Repčík

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np


class EventOfInterest:
    def __init__(
        self,
        event_time: np.ndarray,
        event_magnitude: np.ndarray,
        begin_index: np.integer,
        end_index: np.integer,
        max_index: np.integer,
        free_fall_end_index: np.integer,
    ):
        """
        Basic object, which holds raw data from period of interest and basic information about indexes
        :param event_time: vector of time - seconds
        :param event_magnitude: vector of magnitude - 1D
        :param begin_index: index of the beginning in SensorData
        :param end_index: index of the ending in SensorData
        :param max_index: index of the maximum peak
        :param free_fall_end_index: index of the ending of the free fall
        """
        self.event_time = event_time
        self.event_magnitude = event_magnitude
        self.begin_index = begin_index
        self.end_index = end_index
        self.max_index = max_index
        self.free_fall_end_index = free_fall_end_index

    @property
    def get_from_free_fall(self) -> np.ndarray:
        """
        :return: sometimes we are interested only in the part, when the person hits the ground
        """
        if self.free_fall_end_index is not None:
            return self.event_magnitude[self.free_fall_end_index - self.begin_index :]
        return self.event_magnitude
