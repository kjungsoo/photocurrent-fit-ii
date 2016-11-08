import math


def modified_li2(x):
    accumulator = 0.0
    sign = -1.0
    # I really do not like the way I have implemented this.
    # Hard-coding 50 terms is bogus.
    for idx in range(1, 50):
        n = float(idx)
        accumulator += sign * math.exp(n * x) / (n * n)
        sign *= -1.0

    return accumulator


def f(x):
    return math.pi * math.pi + 3.0 * x * x + 6.0 * modified_li2(x)


def make_photocurrent_func(eta, x0):
    def photocurrent_func(x):
        return eta * f(x - x0)

    return photocurrent_func
