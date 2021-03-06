"""

 Copyright 2021 Chinmaya Srivatsa

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.

"""


import os
import time
import types
from functools import wraps
from typing import Optional, Union, Tuple

import matplotlib.pyplot as plt
import numpy as np


def add_docstring(input_func: Union[types.FunctionType, types.MethodType], doc_to_add: str) -> object:
    """
    Helper function to add docstrings to other functions.
    """

    existing_docstring = ['\n', str(input_func.__doc__).strip(), '\n'] if input_func.__doc__ is not None else []
    input_func.__doc__ = ' '.join(existing_docstring + [doc_to_add])

    return input_func


def return_or_save_figure(func):
    """
    Helper decorator to optionally save a matplotlib figure
    and also optionally return the matplotlib figure for a given plot.
    """

    doc_to_add = \
        """
        Pass save=True as a keyword argument to save figure. 
        
        Pass return_fig=True as a keyword argument to return the figure.
        """

    func = add_docstring(func, doc_to_add)

    @wraps(func)
    def decorator(*args, **kwargs):
        if 'return_fig' in kwargs:
            if not isinstance(kwargs['return_fig'], bool):
                raise TypeError("The 'return_fig' keyword argument only takes boolean values (True/False).")
        else:
            kwargs['return_fig'] = False

        f = func(*args, **kwargs)
        if 'save' in kwargs.keys():
            if not isinstance(kwargs['save'], bool):
                raise TypeError("The 'save' keyword argument only takes boolean values (True/False).")
            elif kwargs['save']:
                plt.savefig(func.__name__)

        return f

    return decorator


def clear_prev_plots(func):
    """
    Helper decorator to remove any previous matplotlib plots before function execution.
    """

    @wraps(func)
    def decorator(*args, **kwargs):
        plt.cla()
        plt.clf()
        plt.close()
        f = func(*args, **kwargs)
        return f

    return decorator


def set_default_labels(func):
    """
    Helper decorator to set default labels to X and Y axes for matplotlib plots.
    """

    @wraps(func)
    def decorator(*args, **kwargs):
        f = func(*args, **kwargs)
        plt.xlabel("X-Axis")
        plt.ylabel("Y-Axis")
        return f

    return decorator


def clear_plots() -> None:
    """
    Helper function to clear all matplotlib plots and figures.
    """

    plt.cla()
    plt.clf()
    plt.close('all')


def timer(func):
    """
    Helper decorator to time any function execution.
    """

    @wraps(func)
    def decorator(*args, **kwargs):
        start = time.perf_counter()
        f = func(*args, **kwargs)
        os.system('echo "' + "Function '{}' took {} seconds".format(func.__name__,
                                                                    round(time.perf_counter() - start, 5)) + '"')
        return f

    return decorator


class DataPointsGenerator:
    """DataPointsGenerator: Generates Data points of different dimensions and distributions."""

    def __init__(self):
        self.__random_state: int = 0

    def set_random_state(self, random_state: int) -> None:
        """
        Sets the random state of the DataPointsGenerator.
        This random state value is used as the seed value in the data point generator functions.

        :param random_state: The value of the the random state.
        """

        if not isinstance(random_state, int):
            raise TypeError("Random State needs to be an Integer between 0 and 2**32 - 1.")
        if random_state < 0:
            raise ValueError("Random State needs to be an Integer between 0 and 2**32 - 1.")

        self.__random_state = random_state

    def __setattr__(self, key, value):
        if key == '__random_state':
            raise AttributeError("The value of random state has already been set. "
                                 "Use the 'set_random_state' method to re-set it to a new value.")
        self.__dict__[key] = value

    def __delattr__(self, key):
        if key == '__random_state':
            raise AttributeError('The random state attribute can not be deleted')
        else:
            del self.__dict__[key]

    def __str__(self):
        return self.__doc__

    def __repr__(self):
        return self.__class__.__name__ + "()"

    @property
    def random_state(self):
        """
        The value of the random state.
        """
        return self.__random_state

    def __set_seed(self) -> None:
        """
        Sets the seed value to be equal the random state.
        This seed value is used in the data point generator functions.
        """
        np.random.seed(seed=self.__random_state)

    def gen_float(self, numrange: Tuple[Union[int, float], Union[int, float]] = (None, None),
                  randomize: Optional[bool] = False) -> float:
        """
        Generates a float between 0 and 1.

        :param numrange: Will set range between which the float is generated.
        :param randomize: Will generate a new float if set to True.
        :return: A float between 0 and 1 excluding 1.
        """

        if not isinstance(numrange, tuple):
            raise TypeError("The range should be a tuple of length 2. ie; (low, high) where low and high "
                            "are integers or floats.")
        if len(numrange) != 2:
            raise ValueError("The range should be a tuple of length 2. ie; (low, high) where low and high "
                             "are integers or floats.")

        if not randomize:
            self.__set_seed()

        if numrange == (None, None):
            return np.random.random()

        return np.random.uniform(numrange[0], numrange[1])

    def gen_normal_1D(self, no_of_points: Optional[int] = 1, randomize: Optional[bool] = False) -> np.ndarray:
        """
        Generates random 1-Dimensional Data.

        :param no_of_points: Number of data points to return.
        :param randomize: Will generate new set of data points if set to True.
        :return: A numpy array of given number of 1-D points or a single 1-D point if not specified.
        """

        if not isinstance(no_of_points, int):
            raise TypeError("Input for number of points needs to be an Integer.")

        if not randomize:
            self.__set_seed()

        return np.random.standard_normal(no_of_points)

    def gen_normal_2D(self, no_of_points: Optional[int] = 1, randomize: Optional[bool] = False) -> np.ndarray:
        """
        Generates random 2-Dimensional Data.

        :param no_of_points: Number of data points to return.
        :param randomize: Will generate new set of data points if set to True.
        :return: A numpy array of given number of 2-D points or a single 2-D point if not specified.
        """

        if not isinstance(no_of_points, int):
            raise TypeError("Input for number of points needs to be an Integer.")

        if not randomize:
            self.__set_seed()

        return np.random.standard_normal(size=(no_of_points, 2))

    def gen_normal_3D(self, no_of_points: Optional[int] = 1, randomize: Optional[bool] = False) -> np.ndarray:
        """
        Generates random 3-Dimensional Data.

        :param no_of_points: Number of data points to return.
        :param randomize: Will generate new set of data points if set to True.
        :return: A numpy array of given number of 3-D points or a single 3-D point if not specified.
        """

        if not isinstance(no_of_points, int):
            raise TypeError("Input for number of points needs to be an Integer.")

        if not randomize:
            self.__set_seed()

        return np.random.standard_normal(size=(no_of_points, 3))

    def gen_linear1D(self, no_of_points: Optional[int] = 1, is_increasing: Optional[bool] = True,
                     randomize: Optional[bool] = False) -> np.ndarray:
        """
        Generates random 1-Dimensional Data in a linearly ascending or descending order.

        :param no_of_points: Shape of the array of points to be returned.
        :param is_increasing: True by default. Returns decreasing, if set to False.
        :param randomize: Will generate new set of data points if set to True.
        :return: A numpy array of given number of 1 Dimensional linearly ascending or descending points.
        """

        if not isinstance(no_of_points, int):
            raise TypeError("Input for number of points needs to be an Integer.")

        if not isinstance(is_increasing, bool):
            raise TypeError("Input for is_increasing needs to be True or False.")

        if not randomize:
            self.__set_seed()

        if not is_increasing:
            return np.array([i + (np.random.uniform(1.5, 3) * np.random.standard_normal())
                             for i in reversed(range(no_of_points))])

        return np.array([i + (np.random.uniform(1.5, 3) * np.random.standard_normal())
                         for i in range(no_of_points)])

    def gen_linear2D(self, no_of_points: Optional[int] = 1, is_increasing: Optional[bool] = True,
                     randomize: Optional[bool] = False) -> np.ndarray:
        """
        Generates random 2-Dimensional Data in a linearly ascending or descending order.

        :param no_of_points: Shape of the array of points to be returned.
        :param is_increasing: True by default. Returns decreasing, if set to False.
        :param randomize: Will generate new set of data points if set to True.
        :return: A numpy array of given number of 2 Dimensional linearly ascending or descending points.
        """

        if not isinstance(no_of_points, int):
            raise TypeError("Input for number of points needs to be an Integer.")

        if not isinstance(is_increasing, bool):
            raise TypeError("Input for is_increasing needs to be True or False.")

        if not randomize:
            self.__set_seed()

        if not is_increasing:
            return np.array([np.array([i + (np.random.uniform(1.5, 3) * np.random.standard_normal()),
                                       j + (np.random.uniform(1.5, 3) * np.random.standard_normal())])
                             for i, j in list(zip(range(no_of_points), reversed(range(no_of_points))))])

        return np.array([np.array([i + (np.random.uniform(1.5, 3) * np.random.standard_normal()),
                                   i + (np.random.uniform(1.5, 3) * np.random.standard_normal())])
                         for i in range(no_of_points)])

    def gen_linear3D(self, no_of_points: Optional[int] = 1, is_increasing: Optional[bool] = True,
                     randomize: Optional[bool] = False) -> np.ndarray:
        """
        Generates random 3-Dimensional Data in a linearly ascending or descending order.

        :param no_of_points: Shape of the array of points to be returned.
        :param is_increasing: True by default. Returns decreasing, if set to False.
        :param randomize: Will generate new set of data points if set to True.
        :return: A numpy array of given number of 3 Dimensional linearly ascending or descending points.
        """

        if not isinstance(no_of_points, int):
            raise TypeError("Input for number of points needs to be an Integer.")

        if not isinstance(is_increasing, bool):
            raise TypeError("Input for is_increasing needs to be True or False.")

        if not randomize:
            self.__set_seed()

        if not is_increasing:
            return np.array([np.array([i + (np.random.uniform(2.5, 5) * np.random.standard_normal()),
                                       i + (np.random.uniform(2.5, 5) * np.random.standard_normal()),
                                       j + (np.random.uniform(2.5, 5) * np.random.standard_normal())])
                             for i, j in list(zip(range(no_of_points), reversed(range(no_of_points))))])

        return np.array([np.array([i + (np.random.uniform(2.5, 5) * np.random.standard_normal()),
                                   i + (np.random.uniform(2.5, 5) * np.random.standard_normal()),
                                   i + (np.random.uniform(2.5, 5) * np.random.standard_normal())])
                         for i in range(no_of_points)])

    @staticmethod
    def gen_line(slope: Union[float, int], intercept: Union[float, int], no_points: Optional[int] = 10) -> np.ndarray:
        """
        Generates a line for a given slope and intercept as y = mx + c

        :param slope: Slope of the line to construct.
        :param intercept: Intercept of the line to construct.
        :param no_points: Number of points to use to construct the line.
        :return: A numpy array of points (x, y) on the line.
        """

        if not isinstance(slope, (float, int)):
            raise TypeError("Slope should be either a float or an integer.")
        if not isinstance(intercept, (float, int)):
            raise TypeError("Intercept should be either a float or an integer.")

        x_vals = np.linspace(1, no_points, no_points)
        y_vals = (slope * x_vals) + intercept

        return np.array(list(zip(x_vals, y_vals)))

    @staticmethod
    def gen_plane(coeff1: Union[float, int], coeff2: Union[float, int], intercept: Union[float, int],
                  no_points: Optional[int] = 10, mesh: Optional[bool] = False) -> np.ndarray:
        """
        Generates a plane for given coefficients and intercept as z = mx1 + nx2 + c

        :param coeff1: First coefficient of the plane to construct.
        :param coeff2: Second coefficient of the plane to construct.
        :param intercept: Intercept of the plane to construct.
        :param no_points: Number of points to use to construct the plane.
        :param mesh: Will return meshgrids to plot surface if set to True.
        :return: A numpy array of 3D points (x1, x2, z) on the plane.
        If mesh=True, a numpy array of meshgrids (X1, X2, Z) to plot the surface of the plane.
        """

        if not isinstance(coeff1, (float, int)):
            raise TypeError("Coefficient 1 should be either a float or an integer.")
        if not isinstance(coeff2, (float, int)):
            raise TypeError("Coefficient 2 should be either a float or an integer.")
        if not isinstance(intercept, (float, int)):
            raise TypeError("Intercept should be either a float or an integer.")

        x1_vals = np.linspace(1, no_points, no_points)
        x2_vals = np.linspace(1, no_points, no_points)

        if mesh:
            X1, X2 = np.meshgrid(x1_vals, x1_vals)
            Z = (coeff1 * X1) + (coeff2 * X2) + intercept
            return np.array((X1, X2, Z))

        z_vals = (coeff1 * x1_vals) + (coeff2 * x2_vals) + intercept

        return np.array(list(zip(x1_vals, x2_vals, z_vals)))

    @staticmethod
    def gen_line_given_x(x_values: Union[np.ndarray, list], slope: Union[float, int],
                         intercept: Union[float, int]) -> np.ndarray:
        """
        Generates a line for a given set of values for x with a slope and intercept as y = mx + c

        :param x_values: Values of x to use to compute y
        :param slope: Slope of the line to construct.
        :param intercept: Intercept of the line to construct.
        :return: A numpy array of points (x, y) on the line.
        """

        if not isinstance(x_values, (np.ndarray, list)):
            raise TypeError("X values should be either a list or a numpy array.")
        if not isinstance(slope, (float, int)):
            raise TypeError("Slope should be either a float or an integer.")
        if not isinstance(intercept, (float, int)):
            raise TypeError("Intercept should be either a float or an integer.")

        y_vals = (slope * np.array(x_values)) + intercept

        return np.array(list(zip(x_values, y_vals)))

    @staticmethod
    def gen_plane_given_x(x1_values: Union[np.ndarray, list], x2_values: Union[np.ndarray, list],
                          coeff1: Union[float, int], coeff2: Union[float, int],
                          intercept: Union[float, int], mesh: Optional[bool] = False) -> np.ndarray:
        """
        Generates a plane for a given set of values for x1 and x2, and coefficients and intercept as z = mx1 + nx2 + c

        :param x1_values: Values of x1 to use to compute Z
        :param x2_values: Values of x2 to use to compute z
        :param coeff1: First coefficient of the plane to construct.
        :param coeff2: Second coefficient of the plane to construct.
        :param intercept: Intercept of the plane to construct.
        :param mesh: Will return meshgrids to plot surface if set to True.
        :return: A numpy array of 3D points (x1, x2, z) on the plane.
        If mesh=True, a numpy array of meshgrids (X1, X2, Z) to plot the surface of the plane.
        """

        if not isinstance(x1_values, (np.ndarray, list)):
            raise TypeError("X1 values should be either a list or a numpy array.")
        if not isinstance(x2_values, (np.ndarray, list)):
            raise TypeError("X2 values should be either a list or a numpy array.")
        if not isinstance(coeff1, (float, int)):
            raise TypeError("Coefficient 1 should be either a float or an integer.")
        if not isinstance(coeff2, (float, int)):
            raise TypeError("Coefficient 2 should be either a float or an integer.")
        if not isinstance(intercept, (float, int)):
            raise TypeError("Intercept should be either a float or an integer.")

        if mesh:
            X1, X2 = np.meshgrid(np.array(x1_values), np.array(x2_values))
            Z = (coeff1 * X1) + (coeff2 * X2) + intercept
            return np.array((X1, X2, Z))

        z_vals = (coeff1 * np.array(x1_values)) + (coeff2 * np.array(x2_values)) + intercept

        return np.array(list(zip(x2_values, x2_values, z_vals)))


if __name__ == '__main__':
    pass
