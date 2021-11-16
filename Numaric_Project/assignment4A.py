"""
In this assignment you should fit a model function of your choice to data 
that you sample from a given function. 

The sampled data is very noisy so you should minimize the mean least squares 
between the model you fit and the data points you sample.  

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You 
must make sure that the fitting function returns at most 5 seconds after the 
allowed running time elapses. If you take an iterative approach and know that 
your iterations may take more than 1-2 seconds break out of any optimization 
loops you have ahead of time.

Note: You are NOT allowed to use any numeric optimization libraries and tools 
for solving this assignment. 

"""

import numpy as np
import time
import random




class Assignment4A:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        solving the assignment for specific functions.
        """

        pass

    def first_iter(self,f,a,b,d):
        Xs = np.random.uniform(a, b, 1000)
        for i in range(d * 2 + 1):
            if i == 0: memo_sigma = np.array([])
            memo_sigma = np.append(memo_sigma, (Xs ** (i)).sum())
        A = np.array([])
        for i in range(d + 1):
            row = np.array([])
            for j in range(d + 1):
                index = i + j
                row = np.append(row, memo_sigma[index])
            A = np.append(A, row)
        A.shape = (d + 1, d + 1)
        Ys = f(Xs)
        b_vec = np.array([])
        for i in range(d + 1):
            b_vec = np.append(b_vec, np.dot(Xs ** i, Ys))
        return memo_sigma, A, b_vec
    def fit(self, f: callable, a: float, b: float, d: int, maxtime: float) -> callable:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape.

        Parameters
        ----------
        f : callable.
            A function which returns an approximate (noisy) Y value given X.
        a: float
            Start of the fitting range
        b: float
            End of the fitting range
        d: int
            The expected degree of a polynomial matching f
        maxtime : float
            This function returns after at most maxtime seconds.

        Returns
        -------
        a function:float->float that fits f between a and b
        """

        first_time = time.time()
        memo_sigma, A, b_vec = self.first_iter(f,a,b,d)
        time_left = maxtime - (time.time() - first_time)
        while time_left > 0.015:
            Xs = np.random.uniform(a, b, int(1700 * time_left))
            if Xs.size < 10: break

            for i in range(d * 2 + 1):
                memo_sigma[i] += (Xs ** (i)).sum()

            for i in range(d + 1):
                for j in range(d + 1):
                    index = i + j
                    A[i, j] = memo_sigma[index]

            Ys = f(Xs)
            for i in range(d + 1):
                b_vec[i] += np.dot(Xs ** i, Ys)


            time_left = maxtime - (time.time() - first_time)

        def lu_matrix_solve(sample_mat):

            n = np.array(sample_mat).shape[0]
            upper = np.zeros((n, n))
            lower = np.zeros((n, n))
            for i in range(n):
                # Upper Triangular
                for k in range(i, n):
                    sigma_upper = 0
                    for j in range(i):
                        sigma_upper += np.multiply(lower[i][j], upper[j][k])
                    # assign values
                    upper[i][k] = sample_mat[i][k] - sigma_upper

                # Lower Triangular
                for k in range(i, n):
                    if (i == k):
                        lower[i][i] = 1
                    else:
                        sigma_lower = 0
                        for j in range(i):
                            sigma_lower += np.multiply(lower[k][j], upper[j][i])

                        lower[k][i] = np.divide((sample_mat[k][i] - sigma_lower), upper[i][i])
            return (lower, upper)

        def solve(sample_mat, vector):
            lower, upper = lu_matrix_solve(sample_mat)
            n = np.array(vector).shape[0]
            for i in range(n):
                for j in range(i):
                    vector[i] -= np.multiply(lower[i, j], vector[j])
            for i in range(n - 1, -1, -1):
                for j in range(i + 1, upper.shape[1]):
                    vector[i] -= np.multiply(upper[i, j], vector[j])
                vector[i] = np.divide(vector[i], upper[i, i])
            return vector

        func_res = (solve(A, b_vec))[::-1]
        return np.poly1d(func_res)



##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment4(unittest.TestCase):

    def test_return(self):
        f = NOISY(0.01)(poly(1,1,1))
        ass4 = Assignment4A()
        T = time.time()
        shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        self.assertLessEqual(T, 5)

    def test_delay(self):
        f = DELAYED(7)(NOISY(0.01)(poly(1,1,1)))

        ass4 = Assignment4A()
        T = time.time()
        shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        self.assertGreaterEqual(T, 5)

    def test_err(self):
        f = poly(1,1,1)
        nf = NOISY(1)(f)
        ass4 = Assignment4A()
        T = time.time()
        ff = ass4.fit(f=nf, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        mse=0
        for x in np.linspace(0,1,1000):
            self.assertNotEqual(f(x), nf(x))
            mse+= (f(x)-ff(x))**2
        mse = mse/1000
        print(mse)


if __name__ == "__main__":
    unittest.main()
