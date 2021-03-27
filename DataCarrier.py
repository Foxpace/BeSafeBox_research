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

from Consts import determine_activity_type, determine_sensor_type
from CustomPaths import test_folder

import os
import pandas as pd
import numpy as np


class DataCarrier:
    def __init__(self, path: str, read_only=None):
        """
        The basic data object for SensorBox a folder with measurements.
        It the processes all the csv files and an extra.txt (not compatible with a new .json file in SensorBox) files.
        :param path: path to folder with the measurements
        :param read_only: list of names of files, which should be read only - None = read all
        """
        self.activity_type = determine_activity_type(
            path
        )  # determines measured activity in folder

        self.annotations_description = {}  # from the extra, mapping for annotations
        self.changes_description = {}  # mapping for the changes.txt file
        self.changes = []  # actual changes
        self.max_values = {}  # max values of the sensor from extra.text
        self.annotations = []  # actual annotations from the extra.txt

        self.sensor_data = {}  # dict for SensorData objects for every sensor
        self.gps_data = None

        self.millis = None
        self.nanos = None
        self.environment = None
        self.holding_position = None

        self.event_holder = None  # place for extracted signal period of interest

        # In the older version of the app, the measurements tented to be split into the multiple files for one sensor.
        # The files of the same type are aggregated into the same list
        file_aggregation = {}

        # iterating through all the files and using specific methods to process them in a correct manner
        for file in [os.path.join(path, f) for f in os.listdir(path)]:

            if "extra" in file:
                self.__process_extra_txt(file)
            elif "confidence" in file:
                self.__process_confidence(file)
            elif "changes" in file:
                self.__process_changes(file)
            elif "GPS" in file:
                self.__process_gps(file)
            elif "csv" in file:
                sensor_type = determine_sensor_type(file)
                if sensor_type is None:
                    continue

                # read only selected files
                if read_only is not None:
                    jump_over = True
                    for specific_file in read_only:
                        if specific_file in file:
                            jump_over = False
                            break
                    if jump_over:
                        continue

                if sensor_type in file_aggregation:
                    file_aggregation[sensor_type].append(file)
                else:
                    file_aggregation[sensor_type] = [file]
            else:
                print("Unknown file {}".format(file))

        # sorts the files of the same sensor and pass them to load them into one dataframe
        for key in file_aggregation.keys():
            paths = file_aggregation[key]
            if len(paths) > 1:
                paths = sorted(
                    paths,
                    key=lambda i: int(
                        os.path.splitext(os.path.basename(i))[0].split("_")[1]
                    ),
                )
            self.__process_sensor_data(paths)

    def __process_extra_txt(self, file_path: str):
        """
        extra.txt has similar formatting as a .fasta format, where keys are stated by ">" and information is below
        HOLD - refers to type of the wearing of the device
        ENVIRONMENT - refers to actual measured activity
        ANNOTATIONS - mapping of custom annotations in device
        MAXVALUES - maximum values of sensors
        Millis - starting time in UNIX time
        Nanos - starting time in nanos of Android system time
        t;c - annotations captured during measurement by user
        :param file_path: path to extra.txt
        """
        with open(file_path, "r", encoding="utf-8") as extra:
            row = extra.readline()
            key = None
            while row:
                if ">" in row or "t;c" in row:  # pick key as beginning
                    key = row
                else:
                    # decides how to read certain information
                    if "HOLD" in key:
                        self.holding_position = row
                    elif "ENVIRONMENT" in key:
                        self.environment = row
                    elif "ANNOTATIONS" in key:
                        if "EMPTY" not in row:
                            annotation_description = row.split(":")
                            self.annotations_description[
                                float(annotation_description[1])
                            ] = annotation_description[0]
                    elif "MAXVALUES" in key:
                        values = row.strip().split(":")
                        self.max_values[values[0]] = float(values[1])
                    elif "Millis" in key:
                        self.millis = float(row.split(":")[1])
                    elif "Nanos" in key:
                        self.nanos = float(row.split(":")[1])
                    elif "t;c" in key:
                        if "x;x" in row:
                            break
                        annotation = row.split(";")
                        self.annotations.append(
                            [float(annotation[0]), float(annotation[1])]
                        )

                row = extra.readline()

    def __process_confidence(self, file_path: str):
        """
        Records of Android activity recognition API
        csv file consists columns with probabilities of certain motions
        For API refer to:
        https://developers.google.com/android/reference/com/google/android/gms/location/ActivityRecognitionClient#requestActivityUpdates(long,%20android.app.PendingIntent)
        :param file_path: path to csv file
        """
        self.confidence = pd.read_csv(file_path, header=0, delimiter=";")

    def __process_changes(self, file_path: str):
        """
        Uses also Android activity recognition API
        In the .txt file, you can find transitions between activities used in API
        for example, user was walking and he entered the car, this can be recognized by API
        firstly, there are numbers of specific activities, which can occur
        after mapping, the recorded transitions with timestamps
        time;actual_type_of_activity
        Refer to:
        https://developers.google.com/android/reference/com/google/android/gms/location/ActivityRecognitionClient#requestActivityUpdates(long,%20android.app.PendingIntent)
        :param file_path: path to txt file
        """
        with open(file_path, "r", encoding="utf-8") as changes:
            row = changes.readline()
            start = False
            while row:
                if "NAME" in row:  # loading mapping
                    start = True
                elif start and ":" in row:
                    c = row.split(":")
                    self.changes_description[c[0]] = int(c[1])
                elif "t;" in row:
                    start = False
                else:
                    c = row.split(";")  # actual records of the transitions
                    self.changes.append(((int(c[0])), int(c[1])))
                row = changes.readline()

    def __process_sensor_data(self, files: list):
        """
        Specific sensor data are stored in the SensorData object
        accessible with t for time, data for numpy array with sensor data and a for accuracy of sample
        t is in nanoseconds (Android system time)
        a is accuracy from 0 to 3 - the lowest accuracy to highest
        :param files: list of files for specific sensor - must be in chronological order (from the oldest to newest)
        """

        sensor_type: str = determine_sensor_type(files[0])
        sensor_data: pd.DataFrame = pd.concat(
            [pd.read_csv(file, delimiter=";") for file in files], ignore_index=True
        )

        time = sensor_data.t.values
        try:
            acc = (
                sensor_data.a.values
            )  # not every sensor has accuracy - e.g. step counter
        except AttributeError:
            acc = None

        temp_data = []
        for column in ["x", "y", "z", "0"]:
            try:
                temp_data.append(
                    sensor_data[column].values
                )  # rotation has 4 axis, pressure only one
            except KeyError:
                break

        # file can be empty, do not store it then
        # sensor data are stacked to one matrix
        if temp_data:
            self.sensor_data[sensor_type] = SensorData(
                time=time, input_data=np.vstack(temp_data), acc=acc
            )

    def __process_gps(self, files: str):
        """
        GPS is stored in one file usually with all coordinates, speed, bearing and accuracy
        :param files: list of files with GPS in it
        """
        self.gps_data = pd.concat([self.gps_data, pd.read_csv(files, delimiter=";")])


class SensorData:
    def __init__(self, time: np.ndarray, input_data: np.ndarray, acc: np.ndarray):
        """
        Simple object to store data
        :param time: time in nanoseconds for most of the sensors
        :param input_data: matrix of raw data
        :param acc: accuracy of samples in matrix
        """
        self.time = time
        self.data = input_data
        self.acc = acc

        # here can be stored modified versions of the data
        # most used Consts.TIME_SECONDS - time converted to seconds, Consts.MAGNITUDE - magnitude of the signal
        self.modified = {}


if __name__ == "__main__":
    data = DataCarrier(test_folder)
