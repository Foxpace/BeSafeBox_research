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

from typing import Optional


class Consts:
    """
    Holds constants, which are used as the keys for accessing different dictionaries
    """

    # types of the motions in which we are interested
    # they are stored in name of the folder
    activity_types = {"SIT", "LAY", "FALL", "WALK", "JUMP", "STAIRS"}
    activity_valid = ["SIT", "LAY", "WALK", "FALL"]
    activity_legend = {"Sit", "Lay", "Walk", "Fall"}

    # placement of the phone in EN, CZ, SK
    hold = [
        ["Pocket", "Kapsa", "Vrecko"],
        ["Handbag", "Kabelka"],
        ["Backpack", "Batoh"],
        ["Holder", "Držák", "Držiak"],
        ["Plain surface", "Volná plocha", "Voľná plocha"],
    ]

    # activities of the person in EN, CZ, SK
    environment = [
        ["Walk", "Chůze", "Chôdza"],
        ["Run", "Běh", "Beh"],
        ["Bike", "Kolo", "Bicykel"],
        ["Public transport", "MHD"],
        ["Train", "Vlak"],
        ["Car / Bus", "Auto / autobus"],
    ]

    # constants for the sensor and list of them
    ACC = "ACC"
    ACG = "ACG"
    AGG = "AGG"
    GYRO = "GYRO"
    MAGNET = "MAGNET"
    ROTATION = "ROTATION"
    PROXI = "PROXI"
    sensor_types = {ACC, ACG, AGG, GYRO, MAGNET, ROTATION, PROXI}

    # modified data
    TIME_SECONDS = "TIME_SECONDS"
    MAGNITUDE = "MAGNITUDE"

    # list of all calculated parameters
    parameters_names = [
        "average",
        "deviation",
        "variance",
        "mobility",
        "complexity",
        "average_tkeo",
        "output",
        "entropy",
        "waveform_length",
        "crest_factor",
        "change_in_angle",
        "change_in_angle_cos",
        "angle_deviation",
        "free fall index",
        "min_max",
        "ratio_3g",
        "kurtosis",
        "skewness",
        "1g_crosses",
    ]

    binary_categories = {0: "Other activity", 1: "Fall"}
    multiple_categories = {0: "Sit", 1: "Lay", 2: "Walk", 3: "Fall"}


def determine_activity_type(path: str) -> Optional[str]:
    """
    finds activity of the folder
    :param path: name of the folder
    :return: measured activity string
    """
    for t in Consts.activity_types:
        if t in path:
            return t
    return None


def determine_sensor_type(path: str) -> Optional[str]:
    """
    Determines type of the file - sensor type
    :param path: path to file
    :return: type of the sensor
    """
    for t in Consts.sensor_types:
        if t in path:
            return t
    return None


def string_activity_to_number(y: str) -> Optional[int]:
    """
    Activity is turned to integer for machine learning purposes
    :param y: string of the activity
    :return: integer of the activity
    """
    if y in Consts.activity_valid:
        return Consts.activity_valid.index(y)
    else:
        return None


def number_to_string_activity_binary(y: int) -> str:
    return Consts.binary_categories[y]


def number_to_string_activity_multiple(y: int) -> str:
    return Consts.multiple_categories[y]
