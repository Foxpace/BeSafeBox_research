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
from numba import njit

from Consts import Consts
from DataCarrier import DataCarrier, SensorData
from EventOfInterest import EventOfInterest


@njit()
def pick_array_of_interest(
    time_seconds, magnitude_vector, threshold_ending=15, begin_max=0.3, end_max=0.7
):
    """
    :param time_seconds: 1D vector
    :param magnitude_vector: 1D vector
    :param threshold_ending: at end, minimal value of ending
    :param begin_max: max time to middle
    :param end_max: max end time of event
    :return: time interval of interesting event (seconds) with magnitude,
    indexes begin, end, max_peak_index, free_fall_end
    """
    max_peak_index = np.int64(np.argmax(magnitude_vector))
    max_peak_time = time_seconds[max_peak_index]
    begin = 0
    end = time_seconds.shape[0]
    free_fall_end = None

    for i in range(max_peak_index, 0, -1):

        # checking for max beginning time
        if abs(max_peak_time - time_seconds[i]) > begin_max:
            begin = i
            break

        # detection of free-fall beneath 0.9g - searching for beginning
        if magnitude_vector[i] <= 9:

            index_free_fall = i
            counter = 0
            free_fall_end = i

            while True:

                # checking for samples higher than 9.25g
                if magnitude_vector[index_free_fall] > 9.25:
                    counter += 1

                    if (
                        counter >= 5
                    ):  # after 5 ticks - add some samples to add some threshold
                        begin = index_free_fall
                        for ii in range(5):
                            if magnitude_vector[index_free_fall + ii] < 20:
                                begin = index_free_fall + ii
                                break
                        break
                    index_free_fall -= 1
                    continue
                else:
                    counter = 0  # reset of ticker if sample is again below 9g

                # checking for max beginning time - max time for free-fall too
                if abs(max_peak_time - time_seconds[index_free_fall]) > begin_max:
                    begin = index_free_fall
                    break

                # event started sooner than measurement
                if index_free_fall == 0:
                    begin = 0
                    break

                index_free_fall -= 1  # subtract index
            break

    # checking for the ending of the event
    top_border = 0

    # adding .7s to ending index
    for i in range(max_peak_index, len(time_seconds)):
        if abs(max_peak_time - time_seconds[i]) > end_max:
            top_border = i
            break

    # searching for first sample with higher amplitude than threshold
    for i in range(top_border, max_peak_index, -1):
        if magnitude_vector[i] > threshold_ending:
            end = i
            break

    # checking for wrong implementation
    if begin > end:
        return None

    if begin < 0:
        begin = 0

    return (
        time_seconds[begin:end],
        magnitude_vector[begin:end],
        begin,
        end,
        max_peak_index,
        free_fall_end,
    )


def get_event_of_interest(
    time_seconds, magnitude_vector, threshold_ending=15, begin_max=0.3, end_max=0.7
) -> EventOfInterest:
    """
    Wrapper for EventOfInterest object and method
    :param time_seconds: 1D vector
    :param magnitude_vector: 1D vector
    :param threshold_ending: at end, minimal value of ending
    :param begin_max: max time to middle
    :param end_max: max end time of event
    :return: EventOfInterest object
    """
    (
        time_seconds,
        magnitude_vector,
        begin,
        end,
        max_peak_index,
        free_fall_end,
    ) = pick_array_of_interest(
        time_seconds=time_seconds,
        magnitude_vector=magnitude_vector,
        threshold_ending=threshold_ending,
        begin_max=begin_max,
        end_max=end_max,
    )
    return EventOfInterest(
        time_seconds, magnitude_vector, begin, end, max_peak_index, free_fall_end
    )


def check_data_integrity_fall_detection(
    data_to_validate: DataCarrier,
    time_threshold=0.2,
    acceleration_threshold=16,
    pick_event=True,
) -> bool:
    """
    Checks if the measurement complies with requirements - checks only acceleration part
    :param acceleration_threshold: magnitude of the measurement is above the threshold
    :param time_threshold: delay between 2 samples is not higher than threshold
    :param data_to_validate: DataCarrier to check
    :param pick_event: if the event of interest should be added to carrier
    :return: boolean if everything is ok
    """
    acceleration: SensorData = data_to_validate.sensor_data[Consts.ACG]
    calculate_time_magnitude(acceleration)

    if np.any(np.diff(acceleration.modified[Consts.TIME_SECONDS]) > time_threshold):
        return False

    if np.all(acceleration.modified[Consts.MAGNITUDE] < acceleration_threshold):
        return False

    if pick_event:
        event_of_interest: EventOfInterest = get_event_of_interest(
            time_seconds=acceleration.modified[Consts.TIME_SECONDS],
            magnitude_vector=acceleration.modified[Consts.MAGNITUDE],
        )

        if event_of_interest is None:
            return False
        else:
            data_to_validate.event_holder = event_of_interest

    return True


def normalize_time(t, conversion_rate=-9) -> np.ndarray:
    """
    converts nanoseconds to seconds
    :param conversion_rate: set -3 different for millis
    :param t: time in nanos
    :return:  time in seconds
    """
    if t is not np.ndarray:
        t = np.array(t)
    return (t - t.item(0)) * (10 ** conversion_rate)


def calculate_time_magnitude(data: SensorData):
    """
    Adds magnitude and time converted to seconds for SensorData object
    :param data: SensorData object
    """
    data.modified[Consts.MAGNITUDE] = np.linalg.norm(data.data, axis=0)
    data.modified[Consts.TIME_SECONDS] = normalize_time(data.time)
