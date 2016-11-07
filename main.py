
from data_loader import get_data

from cost_function import make_cost_func

from data_compiler import compile_data

from partial_derivatives import *

from test_matrix_inversion import invert_2x2

# Our problem is two-dimensional. We have two parameters, eta and x0, that we are trying to determine.
# We need a function that returns a new guess for eta and x0 (as a tuple), and takes the old guess for
# eta and x0.
def new_guess(old_guess, compiled_data):
    cost_func = make_cost_func(compiled_data)
    # First we need the Hessian, which is the matrix of second-order partial derivates.
    a = make_d2_d_eta2(cost_func)(old_guess.eta, old_guess.x0)
    b = make_d2_d_eta_d_x0(cost_func)(old_guess.eta, old_guess.x0)
    c = b
    d = make_d2_d_x02(cost_func)(old_guess.eta, old_guess.x0)

    e, f, g, h = invert_2x2(a, b, c, d)

    r = make_d_d_eta(cost_func)(old_guess.0, old_guess.1)
    s = make_d_d_x0(cost_func)(old_guess.0, old_guess.1)

    # Do the matrix multiplication
    new_guess_0 = old_guess.0 - (e * r + f * s)
    new_guess_1 = old_guess.1 - (g * r + h * s)
    new_guess_tuple = (new_guess_0, new_guess_1)

    return new_guess_tuple


def fit_data(compiled_data_dict):


    print "Fitting " + str(compiled_data_dict)


if __name__ == "__main__":
    stopping_voltages, photocurrents = get_data()
    compiled_data_from_files = compile_data(stopping_voltages, photocurrents)
    fit_data(compiled_data_from_files)
