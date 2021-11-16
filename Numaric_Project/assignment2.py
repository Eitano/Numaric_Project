"""
In this assignment you should find the intersection points for two functions.
"""
from numpy import poly1d
import timeit
import numpy as np
import time
import random
from collections.abc import Iterable
import sympy as sym
import math


class Assignment2:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """
        start_time = timeit.default_timer()
        pass


    def max_err(self, a=0.1, b=10, maxerr=0.001, n=100):
        x = sym.Symbol('x')
        f = 2 ** (1 / x * 2) * sym.sin(1 / x)
        f_1 = f.diff(x)
        f_2 = f_1.diff(x)
        f_3 = f_2.diff(x)
        f_4 = f_3.diff(x)
        f_5 = f_4.diff(x)
        fifth_dev = sym.lambdify(x, f_5)
        fourth_dev = sym.lambdify(x, f_4)
        f2 = lambda x: 0
        roots_arr = self.Range_Newton_Raphson(fifth_dev, f2, a, b, maxerr)
        roots_arr1 = np.append(roots_arr, [a, b])
        values = []
        for i in range(len(roots_arr1)):
            values.append(fourth_dev(roots_arr1[i]))
        best = max(values)
        res = np.multiply(best, (np.divide((b - a) ** 5, (180 * (n) ** 4))))
        return res


    def calculate_diff(self, f1, f2):
        # define new funciton
        g_function = lambda x: f1(x) - f2(x)
        # define the derevaitve
        if type(f1) and type(f2) == poly1d:
            derivative = np.polyder(f1 - f2)
        else:
            derivative = lambda x, h=1e-8: (g_function(x + h) - g_function(x - h)) / (2 * h)

        return (g_function, derivative)


    def Range_Newton_Raphson(self, f1, f2, a, b, maxerr):
        g_function, diff_g = self.calculate_diff(f1, f2)
        roots = []  # the roots we will find - the intercetion between the two funciton

        if maxerr > 0.01:
            step = maxerr
        else:
            step = 0.01

        chunks = np.arange(a,b+step,step)
        for i in range(len(chunks)-1):
            if g_function(chunks[i]) * g_function(chunks[i+1]) < 0 or diff_g(chunks[i]) * diff_g(
                    chunks[i+1]) < 0:
                sample_x = (np.random.rand() * step) + chunks[i]

                for j in range(8):
                    if diff_g(sample_x) == 0:
                        break
                    new_sample_x = sample_x - g_function(sample_x) / diff_g(sample_x)
                    if not (chunks[i] < new_sample_x < chunks[i+1]): break
                    if abs(g_function(new_sample_x)) < maxerr:
                        roots.append(new_sample_x)
                        break
                    sample_x = new_sample_x
        return roots


    def intersections(self, f1: callable, f2: callable, a: float, b: float, maxerr=0.001) -> Iterable:
        """
        Find as many intersection points as you can. The assignment will be
        tested on functions that have at least two intersection points, one
        with a positive x and one with a negative x.
        
        This function may not work correctly if there is infinite number of
        intersection points. 


        Parameters
        ----------
        f1 : callable
            the first given function
        f2 : callable
            the second given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        maxerr : float
            An upper bound on the difference between the
            function values at the approximate intersection points.


        Returns
        -------
        X : iterable of approximate intersection Xs such that for each x in X:
            |f1(x)-f2(x)|<=maxerr.

        """

        return self.Range_Newton_Raphson(f1, f2, a, b,maxerr)


    def max_err(self, a=0.01, b=10, maxerr=0.001, n=100):
        import sympy as sym
        f2 = lambda x: 0
        x = sym.Symbol('x')
        f = 2 ** (1 / x * 2) * sym.sin(1 / x)
        f_1 = f.diff(x)
        f_2 = f_1.diff(x)
        f_3 = f_2.diff(x)
        f_4 = f_3.diff(x)
        f_5 = f_4.diff(x)
        fifth_dev = sym.lambdify(x, f_5)
        fourth_dev = sym.lambdify(x, f_4)
        roots = Assignment2.intersections(fifth_dev, f2, a, b, maxerr)
        roots = np.append(roots, [a, b])
        value_root = []
        for root in roots:
            value_root.append(fourth_dev(root))
        max_value = None
        for num in value_root:
            if (max_value is None or num > max_value):
                max_value = num
        return np.multiply(max_value, ((b - a) ** 5) / (180 * (n) ** 4))

##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment2(unittest.TestCase):

    def test_sqr(self):

        ass2 = Assignment2()

        f1 = np.poly1d([-1, 0, 1])
        f2 = np.poly1d([1, 0, -1])


        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))




    def test_poly(self):

        ass2 = Assignment2()

        f1, f2 = randomIntersectingPolynomials(10)
        X = ass2.intersections(f1, f2, -2, 2, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

if __name__ == "__main__":
    unittest.main()


