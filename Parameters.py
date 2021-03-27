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


from Consts import Consts

from typing import Optional
from numba import njit

import numpy as np
import warnings

from DataCarrier import DataCarrier
from EventOfInterest import EventOfInterest


def basic_stats(event_magnitude: np.ndarray) -> np.ndarray:
    """
    Calculates all the basic statistics from magnitude of the event
    :param event_magnitude: 1D array
    :return: average, standard deviation, variance / activity, mobility, complexity,average tkeo, average output,
    entropy, wavelet length, crest factor
    """
    avg = np.mean(event_magnitude)
    deviation = np.std(event_magnitude)
    variance, mobility, complexity = hjorth_params(event_magnitude)
    tkeo = avg_tkeo(event_magnitude)
    output = avg_output(event_magnitude)
    entropy = ApEn(event_magnitude, 10, 3)
    wave = waveform_length(event_magnitude)
    crest = crest_factor(event_magnitude)
    return np.array(
        [
            avg,
            deviation,
            variance,
            mobility,
            complexity,
            tkeo,
            output,
            entropy,
            wave,
            crest,
        ]
    )


def calculate_acg_parameters(
    time_seconds: np.ndarray,
    magnitude: np.ndarray,
    acg_xyz: np.ndarray,
    event_holder: EventOfInterest,
) -> Optional[np.ndarray]:
    """
    Calculates parameters for given data - basic statistics + parameters specific for acceleration change_in_angle,
    change_in_angle_cos, angle_deviation, free-fall index, min-max, 3g ratio, kurtosis, skewness, 1g crosses

    :param time_seconds: array of time in seconds
    :param magnitude: magnitude of acceleration 1D
    :param acg_xyz: magnitude raw 3D
    :param event_holder: EventHolder object with all the indexes
    :return: basic stats + specific parameters in numpy array
    """
    change_in_angle_value = change_in_angle(acg_xyz)
    change_in_angle_cos_value = change_in_angle_cos(
        time_seconds, acg_xyz, event_holder.begin_index, event_holder.end_index
    )

    if change_in_angle_cos_value is None:
        return None

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        angle_deviation = ad(acg_xyz)
    ffi = free_fall_index(magnitude, event_holder)
    mm = minmax(event_holder.event_magnitude)
    ratio = ratio_3g(event_holder.get_from_free_fall)
    kurt = kurtosis(event_holder.get_from_free_fall)
    skew = skewness(event_holder.get_from_free_fall)
    one_g_crosses = g_cross_rate(event_holder.get_from_free_fall, threshold=9.25)

    return np.append(
        basic_stats(event_holder.event_magnitude),
        np.array(
            [
                change_in_angle_value,
                change_in_angle_cos_value,
                angle_deviation,
                ffi,
                mm,
                ratio,
                kurt,
                skew,
                one_g_crosses,
            ]
        ),
    )


def calculate_acg_parameters_data_carrier(data: DataCarrier) -> Optional[np.ndarray]:
    """
    Calculates same parameters as "calculate_acg_parameters", but uses DataCarrier to export all data
    :param data: DataCarrier with ACG data and with magnitude and time in seconds
    :return: basic statistics + parameters specific for acceleration change_in_angle,
    change_in_angle_cos, angle_deviation, free-fall index, min-max, 3g ratio, kurtosis, skewness, 1g crosses
    """
    return calculate_acg_parameters(
        data.sensor_data[Consts.ACG].modified[Consts.TIME_SECONDS],
        data.sensor_data[Consts.ACG].modified[Consts.MAGNITUDE],
        data.sensor_data[Consts.ACG].data,
        data.event_holder,
    )


def change_in_angle(sensor_values: np.ndarray) -> float:
    """
    authors: SANTOYO-RAMÓN, José Antonio, Eduardo CASILARI and José Manue CANO-GARCÍA
    work: Analysis of a smartphone-based architecture with multiple mobility sensors for fall detection with
    supervised learning.
    DOI: doi:10.3390/s18041155

    Change in angle from X and Z axis of the acceleration - simplified way how to describe change in axis
    :return: change of angle - float
    """
    return np.mean(np.linalg.norm(sensor_values[[0, 2], :], axis=0))


@njit()
def ad(values: np.ndarray) -> float:
    """
    authors: FIGUEIREDO, Isabel N., Carlos LEAL, Luís PINTO, Jason BOLITO a André LEMOS.
    work: Exploring smartphone sensors for fall detection
    DOI: doi:10.1186/s13678-016-0004-1

    Angle deviation - generalization of change in angle, which takes into account all axes
    :param values: angle deviation - float
    """
    summation: float = 0
    passed: int = 0

    previous_norm = np.linalg.norm(values[:, 0])
    for v in range(values.shape[1] - 1):
        new_norm = np.linalg.norm(values[:, v + 1])
        multiplication = previous_norm * new_norm
        previous_norm = new_norm

        dot_product = values[:, v].dot(values[:, v + 1])
        division = dot_product / multiplication
        arc = np.arccos(division)
        # sum of dot product of 2 acceleration samples normalised with multiplication their amplitudes
        # 90° results in nan - ignoring warning, because it is resolved
        if np.isnan(arc):
            passed += 1
        else:
            summation += np.degrees(arc)
    return summation / (float(values.shape[1] - 1 - passed))


def before_and_after_fall(
    time_seconds: np.ndarray,
    acg_xyz: np.ndarray,
    begin_index: np.integer,
    end_index: np.integer,
) -> [np.ndarray, np.ndarray]:
    """
    Picks 1s part of the signal before the main event and after the main event
    :param time_seconds: 1D time series in seconds
    :param acg_xyz: acceleration data 3D
    :param begin_index: beginning of the event
    :param end_index: ending of the event
    :return: 2 arrays with acg samples
    """
    time_begin_before_index = None
    time_end_before_index = begin_index - 1

    time_begin_after_index = end_index + 1
    time_end_after_index = None

    # searching for the beginning
    for i in range(time_end_before_index, 0, -1):
        if np.abs(time_seconds[time_end_before_index] - time_seconds[i]) >= 1:
            time_begin_before_index = i
            break

    if time_begin_before_index is None:
        time_begin_before_index = 0

    # searching for the end
    for i in range(time_begin_after_index, len(time_seconds)):
        if np.abs(time_seconds[time_begin_after_index] - time_seconds[i]) >= 1:
            time_end_after_index = i
            break

    if time_begin_before_index is None:
        time_begin_before_index = 0
    if time_end_after_index is None:
        time_end_after_index = len(time_seconds)

    return (
        acg_xyz[:, time_begin_before_index:time_end_before_index],
        acg_xyz[:, time_begin_after_index:time_end_after_index],
    )


def change_in_angle_cos(
    time_seconds: np.ndarray,
    acg_xyz: np.ndarray,
    begin_index: np.integer,
    end_index: np.integer,
):
    """
    authors: FIGUEIREDO, Isabel N., Carlos LEAL, Luís PINTO, Jason BOLITO a André LEMOS.
    work: Exploring smartphone sensors for fall detection
    DOI: doi:10.1186/s13678-016-0004-1

    similar to change in angle, but it is taken from 1s before fall and 1s after fall
    :param time_seconds: time in seconds
    :param acg_xyz: acceleration data
    :param begin_index: beginning index of event
    :param end_index: ending index of event
    :return:float in degrees
    """

    # picking one second before and after the event
    values_xyz_before, values_xyz_after = before_and_after_fall(
        time_seconds, acg_xyz, begin_index, end_index
    )
    if len(values_xyz_before[0]) == 0 or len(values_xyz_after[0]) == 0:
        return None

    aa: np.ndarray = np.average(values_xyz_before, axis=1)
    ab: np.ndarray = np.average(values_xyz_after, axis=1)

    acca: np.float64 = np.linalg.norm(aa)
    accb: np.float64 = np.linalg.norm(ab)

    return np.degrees(np.arccos((aa.dot(ab)) / (acca * accb)))


def free_fall_index(values_acg: np.array, event_of_interest: EventOfInterest) -> float:
    """
    authors: ABBATE, Stefano, Marco AVVENUTI, Guglielmo COLA, Paolo CORSINI, Janet LIGHT a Alessio VECCHIO.
    work: Recognition of false alarms in fall detection systems
    DOI: doi:10.1109/CCNC.2011.5766464

    Average value of dip before impact
    :param values_acg: acceleration data
    :param event_of_interest: object
    :return: 10 if dip does not exists, lower number otherwise
    """
    if event_of_interest.free_fall_end_index is None:
        return 10.0
    return np.mean(
        values_acg[
            event_of_interest.begin_index : event_of_interest.free_fall_end_index
        ]
    )


def minmax(magnitude: np.ndarray) -> float:
    """
    difference minimum between maximum value
    :param magnitude: 1D array of magnitude
    :return: float
    """
    return np.max(magnitude) - np.min(magnitude)


def ratio_3g(magnitude: np.ndarray, threshold=30) -> float:
    """
    Ratio of samples above 3g and below
    :param magnitude: acceleration
    :param threshold: can be customized
    :return: float with ratio
    """
    return np.sum(magnitude > threshold) / np.sum(magnitude < threshold)


def kurtosis(magnitude: np.ndarray) -> float:
    """
    Kurtosis of the magnitude - 4. standardized moment
    resource:
    https://mathworld.wolfram.com/StandardizedMoment.html
    https://en.wikipedia.org/wiki/Standardized_moment
    :param magnitude: 1D magnitude of acceleration
    :return: kurtosis float
    """
    return momentum(magnitude, 4) / np.power(momentum(magnitude, 2), 2)


def skewness(magnitude: np.ndarray) -> float:
    """
    Skewness of the magnitude - 3. standardized moment
    resource:
    https://mathworld.wolfram.com/StandardizedMoment.html
    https://en.wikipedia.org/wiki/Standardized_moment
    :param magnitude: 1D magnitude of acceleration
    :return: kurtosis float
    """
    return momentum(magnitude, 3) / np.power(momentum(magnitude, 2), 1.5)


def momentum(magnitude: np.ndarray, moment=2) -> float:
    """
    Moment calculation for skewness and kurtosis
    resource:
    https://mathworld.wolfram.com/StandardizedMoment.html
    https://en.wikipedia.org/wiki/Standardized_moment
    :param magnitude: 1D array of acceleration
    :param moment: degree of moment 2,3,4,...
    :return: moment as float
    """
    avg = np.mean(magnitude)
    return np.mean([np.power(m - avg, moment) for m in magnitude])


def hjorth_params(magnitude: np.ndarray) -> np.ndarray:
    """
    author: HJORT, Bo
    work: EEG Analysis Based On Time Domain Properties
    DOI: doi:10.1016/0013-4694(70)90143-4

    Calculates basic Hjorth descriptors
    :param magnitude: of acceleration
    :return: array of activity, mobility, complexity
    """
    activity = np.var(magnitude)

    d1 = np.diff(magnitude)
    mobility = np.sqrt(np.var(d1) / activity)

    d2 = np.diff(d1)
    complexity = np.sqrt(np.var(d2) / mobility)

    return np.array([activity, mobility, complexity])


def avg_tkeo(magnitude: np.ndarray) -> float:
    """
    author: MARAGOS, P., J.F. KAISER a T.F. QUATIERI
    work: On amplitude and frequency demodulation using energy operators
    DOI: doi:10.1109/78.212729

    Calculates average TKEO coeficient for measurement
    :param magnitude: 1D acceleration as magnitude
    :return: averaged TKEO as float
    """
    return np.sum(
        [
            np.power(magnitude[i], 2) + (magnitude[i - 1] * magnitude[i + 1])
            for i in range(1, magnitude.shape[0] - 1)
        ]
    ) / (float(magnitude.shape[0]) - 2)


def avg_output(magnitude: np.ndarray) -> float:
    """
    Power of 2 for whole input, which is averaged
    :param magnitude: acceleration magnitude
    :return: averaged float
    """
    return np.mean([np.power(i, 2) for i in magnitude])


def ApEn(magnitude: np.ndarray, m: int, r: float) -> float:
    """
    An approximate entropy (ApEn) is a technique used to quantify the amount of regularity and the
    unpredictability of fluctuations over time-series data
    Resource: https://en.wikipedia.org/wiki/Approximate_entropy
    :param magnitude: 1D array of values
    :param m: length of compared data
    :param r: filtering level
    :return: entropy as float
    """
    #

    def __maxdist(x_i, x_j):
        return np.max([np.abs(ua - va) for ua, va in zip(x_i, x_j)])

    def __phi(m: int):
        x = [[magnitude[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [
            len([1 for x_j in x if __maxdist(x_i, x_j) <= r]) / (N - m + 1.0)
            for x_i in x
        ]
        nm = N - m
        if nm == 0:
            nm = 1

        return nm ** (-1) * np.sum(np.log(C))

    if r <= 0:
        raise ValueError("Must be positive real number")
    N = magnitude.shape[0]

    return np.abs(__phi(m + 1) - __phi(m)) / N


def waveform_length(amplitude: np.ndarray) -> float:
    """
    First differential with absolute value, which is averaged through whole signal
    :param amplitude: 1D array
    :return: waveform length - float
    """
    return np.average(np.abs(np.diff(amplitude)))


def crest_factor(amplitude: np.ndarray) -> float:
    """
    Peak value / RMS - root mean square - comparison of peak value to RMS
    resource:  https://tmi.yokogawa.com/library/resources/training-modules/power-meter-tutorials-background/
    :param amplitude: 1D array
    :return: crest factor float
    """
    return np.abs(np.max(amplitude)) / np.sqrt(
        np.sum(amplitude ** 2) / amplitude.shape[0]
    )


def g_cross_rate(magnitude, threshold=9.25) -> int:
    """
    Number of crosses through specific threshold
    :param magnitude: 1D array
    :param threshold: to follow
    :return: integer
    """
    crossed = False
    crosses = 0
    for i in magnitude:
        if crossed and i > threshold:
            crosses += 1
            crossed = False
        elif crossed is False and i < threshold:
            crosses += 1
            crossed = True
    return crosses


def moving_average(a, n=3) -> np.ndarray:
    """
    Moving average with specified window
    :param a: 1D array
    :param n: size of window
    :return: average of the signal
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n
