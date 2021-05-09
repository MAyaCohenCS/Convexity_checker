from inspect import signature
from typing import Callable, Tuple
import numpy as np
import sys


delta = 1e-4
epsilon = 1e-6  # for float comparison
sample_range = (-1., 1)
test_sample_size = 100


def _derive(f: Callable, x: np.array, coord: int) -> float:
    shifted_x = np.copy(x)
    shifted_x[coord] += delta

    return (f(*shifted_x) - f(*x)) / delta


def _sample_inputs(f: Callable, inputs_num: int = 1, sample_range: Tuple[float, float] = sample_range):
    sig = signature(f)
    return np.random.uniform(-1, 1, (inputs_num, len(sig.parameters)))


def _sample_thetas(num_of_samples=1):
    return np.random.uniform(0, 1, num_of_samples)


def gradient(f: Callable, x: np.array) -> Callable:
    grad = np.zeros(x.size)

    for i in range(x.size):
        grad[i] = _derive(f, x, i)

    return grad


def hessian(f: Callable, x: np.array):
    hess = np.zeros((x.size, x.size))
    for i in range(x.size):
        for j in range(x.size):
            df = lambda *args: _derive(f, args, i)
            hess[i, j] = _derive(df,x, j)

    return hess


def is_convex(f: Callable, test_size=test_sample_size):
    xs = _sample_inputs(f, test_size)
    ys = _sample_inputs(f, test_size)
    thetas = _sample_thetas(test_size)

    for i in range(test_size):
        t = thetas[i]
        x = xs[i]
        y = ys[i]

        func = f(*(x * t + y * (1 - t)))
        line = f(*x) * t + f(*y) * (1 - t)
        conv_condition = func <= line + epsilon

        if not conv_condition:
            return False
    return True


def is_convex_1st_order(f: Callable, test_size=test_sample_size):
    xs = _sample_inputs(f, test_size)
    ys = _sample_inputs(f, test_size)

    for i in range(test_size):
        x = xs[i]
        y = ys[i]

        fx = f(*x)
        tangent = f(*y) + gradient(f, y) @ (x-y)
        conv_condition = (fx + epsilon) >= tangent

        if not conv_condition:
            return False
    return True

def is_convex_2nd_order(f: Callable, test_size=test_sample_size):
    xs = _sample_inputs(f, test_size)
    zs = _sample_inputs(f, test_size)

    for i in range(test_size):
        x = xs[i]
        z = zs[i] + epsilon

        hess = hessian(f, x)
        quad_random_vector = z @ hess @ z
        conv_condition = quad_random_vector + epsilon >= 0

        if not conv_condition:
            return False
    return True


def k(x1, x2):  # debug func
    return x1**2


def h(x1: float, x2: float) -> float: # debug func
    return x2 * x1


def g(x1,x2,x3,x4): # debug func
    return x1 + x2 + 2*x3


def f(x1, x2, x3, x4):
    exp1 = x1**4 * x2
    exp2 = x3/np.square(1+x2)
    exp3 = np.exp(x3) * x1 * 100
    exp4 = x4**3

    return exp1 - exp2 + exp3 + exp4

if __name__ == '__main__':

    sys.stdout = open("convexity check results.txt", "w")

    test_func = f

    print('-' * 30)
    print(f'Sample a point and print gradient and hessian in it:')
    print('-'*30)
    sample_points = _sample_inputs(test_func, 1)

    print(f'Sampled point: {sample_points[0]}')
    print(f'gradient: {gradient(test_func, sample_points[0])}')
    print(f'Hessian: \n{hessian(test_func, sample_points[0])}')

    print('-' * 30)
    print(f'is convex 0 order: {is_convex(test_func)}')
    print(f'is convex 1 order: {is_convex_1st_order(test_func)}')
    print(f'is convex 2 order: {is_convex_2nd_order(test_func)}')

    sys.stdout.close()
