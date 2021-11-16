"""
In this assignment you should find the area enclosed between the two given functions.
The rightmost and the leftmost x values for the integration are the rightmost and 
the leftmost intersection points of the two functions. 

The functions for the numeric answers are specified in MOODLE. 


This assignment is more complicated than Assignment1 and Assignment2 because: 
    1. You should work with float32 precision only (in all calculations) and minimize the floating point errors. 
    2. You have the freedom to choose how to calculate the area between the two functions. 
    3. The functions may intersect multiple times. Here is an example: 
        https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx
    4. Some of the functions are hard to integrate accurately. 
       You should explain why in one of the theoretical questions in MOODLE. 

"""
import numpy as np
import time
import random
from assignment2 import Assignment2


class Assignment3:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass


    def integrate(self, f: callable, a: float, b: float, n: int) -> np.float32:
        """
        Integrate the function f in the closed range [a,b] using at most n 
        points. Your main objective is minimizing the integration error. 
        Your secondary objective is minimizing the running time. The assignment
        will be tested on variety of different functions. 
        
        Integration error will be measured compared to the actual value of the 
        definite integral. 
        
        Note: It is forbidden to call f more than n times. 
        
        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the integration range.
        b : float
            end of the integration range.
        n : int
            maximal number of points to use.

        Returns
        -------
        np.float32
            The definite integral of f between a and b
        """

        oddSum = np.float32()
        evenSum = np.float32()
        width = np.divide((b - a), (2 * n))
        # calculte even sum
        for i in range(1, n):
            oddSum += f(2 * width * i + a)
        for i in range(1, n + 1):
            evenSum += f(width * (-1 + 2 * i)+ a)
        summraize = np.multiply(np.sum([f(a), f(b), oddSum * 2, evenSum * 4]) / 3, width)
        return np.float32(summraize)



    def roots_area_between(self, root_arr, h):

        root_plus_Zero = np.append(np.array([0]), root_arr)
        sum_area = np.float32()
        for i in range(len(root_plus_Zero)-1):
            sum_area += abs(self.integrate(h, root_plus_Zero[i], root_plus_Zero[i+1], n=100))
        return abs(sum_area)


    def areabetween(self, f1: callable, f2: callable) -> np.float32:
        """
        Finds the area enclosed between two functions. This method finds 
        all intersection points between the two functions to work correctly. 
        
        Example: https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx

        Note, there is no such thing as negative area. 
        
        In order to find the enclosed area the given functions must intersect 
        in at least two points. If the functions do not intersect or intersect 
        in less than two points this function returns NaN.  
        This function may not work correctly if there is infinite number of 
        intersection points. 
        

        Parameters
        ----------
        f1,f2 : callable. These are the given functions

        Returns
        -------
        np.float32
            The area between function and the X axis

        """

        # replace this line with your solution
        roots_array = Assignment2().intersections(f1=f1, f2=f2, a=1, b=100, maxerr=0.001)
        res = self.roots_area_between(roots_array, lambda x:f1(x) -f2(x))
        return np.float32(res)



    # def max_err(self, a=0.1, b=10, maxerr=0.001, n=100):
    #     import sympy as sym
    #     x = sym.Symbol('x')
    #     f = 2 ** (1 / x * 2) * sym.sin(1 / x)
    #     f_1 = f.diff(x)
    #     f_2 = f_1.diff(x)
    #     f_3 = f_2.diff(x)
    #     f_4 = f_3.diff(x)
    #     f_5 = f_4.diff(x)
    #     fifth_dev = sym.lambdify(x, f_5)
    #     fourth_dev = sym.lambdify(x, f_4)
    #     f2 = lambda x: 0
    #     roots_arr = Assignment2.intersections(fifth_dev, f2, a, b, maxerr)
    #     roots_arr = np.append(roots_arr, [a, b])
    #     values = []
    #     for i in range(len(roots_arr)):
    #         values.append(fourth_dev(roots_arr[i]))
    #     max_value = None
    #     for num in values:
    #         if (max_value is None or num > max_value):
    #             max_value = num
    #     res = np.multiply(max_value, ((b - a) ** 5) / (180 * (n) ** 4))
    #     return res



##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment3(unittest.TestCase):

    def test_integrate_float32(self):
        ass3 = Assignment3()
        f1 = np.poly1d([-1, 0,5])
        f5 = lambda x: np.sin(5 * x)
        # r = ass3.integrate(f1, -1, 1, 10)
        res =ass3.areabetween(f1,f5)
        g = res

        # self.assertEquals(r.dtype, np.float32)

    def test_integrate_hard_case(self):
        ass3 = Assignment3()
        f1 = strong_oscilations()
        r = ass3.integrate(f1, 0.09, 10, 20)
        true_result = -7.78662 * 10 ** 33
        self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))


if __name__ == "__main__":
    unittest.main()


