
import collections

from data_loader import get_data

from cost_function import make_cost_func

from data_compiler import compile_data

from partial_derivatives import *

from test_matrix_inversion import invert_2x2

import math  # for sqrt

GuessVector = collections.namedtuple('GuessVector', 'eta V0')


# Our problem is two-dimensional. We have two parameters, eta and x0, that we are trying to determine.
# We need a function that returns a new guess for eta and x0 (as a tuple), and takes the old guess for
# eta and x0.
def new_guess(old_guess, compiled_data):
    cost_func = make_cost_func(compiled_data)
    # First we need the Hessian, which is the matrix of second-order partial derivatives.
    a = make_d2_d_eta2(cost_func)(old_guess.eta, old_guess.V0)
    b = make_d2_d_eta_d_x0(cost_func)(old_guess.eta, old_guess.V0)
    c = b
    d = make_d2_d_x02(cost_func)(old_guess.eta, old_guess.V0)

    inverted_hessian = invert_2x2(a, b, c, d)

    e = inverted_hessian[0]
    f = inverted_hessian[1]
    g = inverted_hessian[2]
    h = inverted_hessian[3]

    r = make_d_d_eta(cost_func)(old_guess.eta, old_guess.V0)
    s = make_d_d_x0(cost_func)(old_guess.eta, old_guess.V0)

    # Do the matrix multiplication
    new_guess_vector = GuessVector(eta=old_guess.eta - (e * r + f * s), V0=old_guess.V0 - (g * r + h * s))

    return new_guess_vector


def compute_convergence_distance(new_guess_vector, old_guess_vector):
    delta_vector = GuessVector(eta=new_guess_vector.eta - old_guess_vector.eta,
                               V0=new_guess_vector.V0 - old_guess_vector.V0)

    convergence_distance = math.sqrt(delta_vector.eta * delta_vector.eta + delta_vector.V0 * delta_vector.V0)
    return convergence_distance


def fit_data(compiled_data_dict):

    epsilon = 0.0001
    old_guess_vector = GuessVector(1.0, 0.0)

    # Make the initial guess
    new_guess_vector = new_guess(old_guess_vector, compiled_data_dict)
    while compute_convergence_distance(new_guess_vector, old_guess_vector) > epsilon:
        # guess again
        old_guess_vector = new_guess_vector
        new_guess_vector = new_guess(old_guess_vector, compiled_data_dict)

    print "Result for eta = " + new_guess_vector.eta + " and V0 = " + new_guess_vector.V0


if __name__ == "__main__":
    stopping_voltages, photocurrents = get_data()
    compiled_data_from_files = compile_data(stopping_voltages, photocurrents)
    fit_data(compiled_data_from_files)
