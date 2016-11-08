
import unittest

from partial_derivatives import make_d_d_eta


def rock_paper_scissors(eta, x0):
    return 0.5 * eta * eta + x0 * x0 + 0.5 * eta * eta * x0 * x0


class TestPartialDerivatives(unittest.TestCase):

    # Test the partial derivative code
    def broken_test_take_d_d_eta(self):

        d_d_eta = make_d_d_eta(rock_paper_scissors)

        # We are expecting this to be eta + eta * x0 * x0, approximately.

        eta = 0.5
        x0 = 2.0

        result = d_d_eta(eta, x0)

        self.assertEquals(4.5, result)
