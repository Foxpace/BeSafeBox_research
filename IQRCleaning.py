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

import pandas as pd
import numpy as np
from tqdm.notebook import tqdm


def get_fences(df: pd.DataFrame) -> dict:
    """
    Gets fences for indication of outliers by IQR rule
    :param df: dataframe of features
    :return: fences in dictionary by column names - (lower, higher fence) tuple
    """
    fences = {}
    for feature in df.columns:
        q1 = df[feature].quantile(0.25)
        q3 = df[feature].quantile(0.75)
        iqr = q3 - q1
        upper_fence = q3 + 2 * iqr
        lower_fence = q1 - 2 * iqr
        fences[feature] = (lower_fence, upper_fence)
    return fences


def iqr_rule(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creation of fences based on inter quartile rule - every number beyond fence, is replaced with border number
    :param df: any number dataframe
    :return: modified dataframe by IQR rule
    """
    result: pd.DataFrame = df.copy()
    fences = get_fences(result)
    for feature in df.columns:
        result[feature] = result[feature].apply(
            lambda x: x if x <= fences[feature][1] else fences[feature][1]
        )
        result[feature] = result[feature].apply(
            lambda x: x if x >= fences[feature][0] else fences[feature][0]
        )
    return result


def iqr_rule_outliers(
    df: pd.DataFrame, categories, number_of_neighbours=5, apply_iqr=False
) -> [np.ndarray, dict]:
    """
    Tries to find the closest neighbours to every row and average them, if some parameter in the row has the
    higher or lower value than fence respectively. Outlier is detected by the IQR rule. If the parameter is still beyond
    the fence after averaging, the value stays the same. Rows of the same category are only used.
    :param apply_iqr: applied IQR rule to move values beyond fences to the border value
    :param number_of_neighbours: number of the rows to use for the new value
    :param df: parameters in a dataframe in columns
    :param categories:
    :return: cleaned numpy matrix with found fences for categories in dataframe
    """
    result: pd.DataFrame = df.copy()
    matrix = result.values
    fences = get_fences(result)

    for feature_number, feature in enumerate(
        tqdm(df.columns, desc="Processed features")
    ):
        values = result[feature].values
        for i, value in enumerate(values):

            # comparison with fences - change only value over or under fences
            if value > fences[feature][1] or value < fences[feature][0]:

                actual_category = categories[i]  # get type
                indexes_actual_category = np.where(categories == actual_category)[
                    0
                ]  # use only rows of same type
                # remove itself
                indexes_actual_category = np.delete(
                    indexes_actual_category, np.where(indexes_actual_category == i)[0]
                )
                picked = matrix[indexes_actual_category, :]
                to_subtract = matrix[i, :]
                subtract = picked - to_subtract
                summed = np.sum(subtract, axis=1)
                sub = np.abs(summed)  # subtract from matrix

                closest = np.argsort(sub)[
                    :number_of_neighbours
                ]  # find the closest ones

                # average values of closest neighbours
                new_value = np.average(
                    matrix[indexes_actual_category[closest]][:, feature_number]
                )

                # # ignore if the value is still beyond the fences
                if new_value > fences[feature][1] or new_value < fences[feature][0]:
                    continue

                values[i] = new_value

        result[feature] = values

        if apply_iqr:
            result = iqr_rule(result)

    return result, fences
