"""
In this assignment you should interpolate the given function.
"""

import numpy as np
import time
import random


class Assignment1:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        starting to interpolate arbitrary functions.
        """

        pass

    def interpolate(self, f: callable, a: float, b: float, n: int) -> callable:
        """
        Interpolate the function f in the closed range [a,b] using at most n 
        points. Your main objective is minimizing the interpolation error.
        Your secondary objective is minimizing the running time. 
        The assignment will be tested on variety of different functions with 
        large n values. 
        
        Interpolation error will be measured as the average absolute error at 
        2*n random points between a and b. See test_with_poly() below. 

        Note: It is forbidden to call f more than n times. 

        Note: This assignment can be solved trivially with running time O(n^2)
        or it can be solved with running time of O(n) with some preprocessing.
        **Accurate O(n) solutions will receive higher grades.** 
        
        Note: sometimes you can get very accurate solutions with only few points, 
        significantly less than n. 
        
        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        n : int
            maximal number of points to use.

        Returns
        -------
        The interpolating function.
        """
        ########################## n=1

        list_x = np.linspace(a, b, n)
        list_y = np.array([f(x) for x in list_x])
        vector_k=np.empty(n-1)
        vector_k.fill(0)
        vector_k[0] = list_y[0] + 2 * list_y[1]
        for i in range(1, n - 1):
            vector_k[i] = 4 * list_y[i] + 2 * list_y[i + 1]
        vector_k[-1] = 8 * list_y[-2] + list_y[-1]

        a_vector = np.array(tridiag_solve(vector_k,n))

        b_vector = np.empty(n - 1)
        b_vector.fill(0)
        for i in range(n - 2):
            b_vector[i] = 2 *list_y[i + 1] - a_vector[i + 1]
        b_vector[n - 2] = (a_vector[n - 2] + list_y[n - 1]) / 2

        def h(x):
            index=np.where(list_x==min(list_x,key=lambda x_in_list: abs(x-x_in_list)))
            if (x>=list_x[index[0][0]]):
                control=(x-list_x[index[0][0]])/(list_x[index[0][0]+1]-list_x[index[0][0]])
            else:
                control=(x-list_x[index[0][0]-1])/(list_x[index[0][0]]-list_x[index[0][0]-1])
                index=index[0]-1
            to_return = lambda t: np.power((1 - t), 3) * list_y[index[0]] + 3 * a_vector[
                index[0]] * t * np.power((1 - t), 2) + 3 * np.power(t, 2) * (1 - t) * b_vector[index[0]] + list_y[
                                         index[0] + 1] * np.power(t, 3)
            return to_return(control)
        return h

        # replace this line with your solution to pass the second test
        result = lambda x:x;

        return result

def tridiag_solve(k,n):
    triang_1=np.empty(n-2,dtype=float)
    triang_1.fill(1)
    triang_1[-1] = 2
    triang_1_copy = np.copy(triang_1)

    triang_2=np.empty(n-1)
    triang_2.fill(4)
    triang_2[0] = 2
    triang_2[-1] = 7
    triang_2_copy = np.copy(triang_2)

    triang_3=np.empty(n-2)
    triang_3.fill(1)
    triang_3_copy = np.copy(triang_3)

    k_copy = np.copy(k)

    for i in range(1, len(k)):
        mc = triang_1_copy[i - 1] / triang_2_copy[i - 1]
        triang_2_copy[i] = triang_2_copy[i] - mc * triang_3_copy[i - 1]
        k_copy[i] = k_copy[i] - mc * k_copy[i - 1]

    a_vector = list(triang_2_copy)
    a_vector[-1] = k_copy[-1] / triang_2_copy[-1]

    for i in range(len(k) - 2, -1, -1):
        a_vector[i] = (k_copy[i] - triang_3_copy[i] * a_vector[i + 1]) / triang_2_copy[i]
    return a_vector

##########################################################################


import unittest
from functionUtils import *
from tqdm import tqdm


class TestAssignment1(unittest.TestCase):

    def test_with_poly(self):

        T = time.time()

        ass1 = Assignment1()
        mean_err = 0

        d = 30
        for i in tqdm(range(100)):
            a = np.random.randn(d)

            f = np.poly1d(a)
            ff = ass1.interpolate(f, -10, 10, 100)

            xs = np.random.random(200)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / 200
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print(T)
        print(mean_err)

    def test_with_poly_restrict(self):
        ass1 = Assignment1()
        a = np.random.randn(5)
        f = RESTRICT_INVOCATIONS(10)(np.poly1d(a))
        ff = ass1.interpolate(f, -10, 10, 10)
        xs = np.random.random(20)
        for x in xs:
            yy = ff(x)

if __name__ == "__main__":
    unittest.main()


