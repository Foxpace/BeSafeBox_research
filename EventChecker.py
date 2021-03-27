from typing import Optional

import numpy as np


class EventOfInterest:
    def __init__(
        self,
        event_time: np.ndarray,
        event_magnitude: np.ndarray,
        begin_index: int,
        end_index: int,
        max_index: int,
        free_fall_end_index: int,
    ):
        self.event_time = event_time
        self.event_magnitude = event_magnitude
        self.begin_index = begin_index
        self.end_index = end_index
        self.max_index = max_index
        self.free_fall_end_index = free_fall_end_index

    def get_from_free_fall(self, amplitude_vector: np.ndarray):
        if self.free_fall_end_index is not None:
            return amplitude_vector[self.free_fall_end_index - self.begin_index:]
        return amplitude_vector[self.begin_index:self.end_index]


def pick_array_of_interest(
    time_seconds, magnitude_vector, threshold_ending=15, begin_max=0.3, end_max=0.7
) -> Optional[EventOfInterest]:
    """

    :param time_seconds: time of array
    :param magnitude_vector: self explanatory
    :param threshold_ending: at end, value of ending
    :param begin_max: max time from middle
    :param end_max: max end time
    :return: ArrayOfInterest object, which concludes everything
    """
    max_peak_index = int(np.argmax(np.array(magnitude_vector)))
    max_peak_time = time_seconds[max_peak_index]
    begin = 0
    end = int(len(time_seconds))
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

    return EventOfInterest(
        time_seconds[begin:end],
        magnitude_vector[begin:end],
        begin,
        end,
        max_peak_index,
        free_fall_end,
    )
