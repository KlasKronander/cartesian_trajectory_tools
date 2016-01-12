import numpy as np
from matplotlib import pyplot as plt

class PolynomialInterpolator(object):
    def __init__(self, x, y, yd=None, ydd=None):
        # TODO: asssert increasing order of x! assert shape!
        yd = np.array([0.0, 0.0]) if yd is None else yd
        ydd = np.array([0.0, 0.0]) if ydd is None else ydd
        if not (yd.shape == y.shape and ydd.shape == y.shape and x.shape == y.shape):
            raise ValueError("You have provided input data with incoherent sizes")
        A = np.array([
            [x[0]**5, x[0]**4, x[0]**3, x[0]**2, x[0], 1.],
            [5*x[0]**4, 4*x[0]**3, 3*x[0]**2, 2*x[0], 1., 0.],
            [20*x[0]**3, 12*x[0]**2, 6*x[0], 2.0, 0.0, 0.0],
            [x[1]**5, x[1]**4, x[1]**3, x[1]**2, x[1], 1.],
            [5*x[1]**4, 4*x[1]**3, 3*x[1]**2, 2*x[1], 1., 0.],
            [20*x[1]**3, 12*x[1]**2, 6*x[1], 2.0, 0.0, 0.0],
        ])
        b = np.array([y[0], yd[0], ydd[0], y[1], yd[1], ydd[1]])
        self.x = x
        self.y = y
        self.coefficients = np.linalg.solve(A,b)

    def evaluate_polynomial(self, x_q):
        return np.polyval(self.coefficients, x_q)

    def __call__(self, x_q):
        y_q = np.polyval(self.coefficients, x_q)
        y_q[np.where(x_q < self.x[0])] = self.y[0]
        y_q[np.where(x_q > self.x[1])] = self.y[1]
        return y_q


class ViaPointInterpolator(object):
    def __init__(self, x, y, yd=None, ydd=None):
        if any(np.sort(x) != x):
            raise Exception("x was not provided in increasing order!")
        n_interpolators = x.shape[0] - 1  # one fewer interpolator than points
        self.interpolators = []
        self.x = x
        self.y = y
        # need to prepare yd!!
        yd = np.zeros(x.shape) if yd is None else yd
        ydd = np.zeros(x.shape) if ydd is None else ydd
        if not (yd.shape == y.shape and x.shape == y.shape and ydd.shape == y.shape):
            raise ValueError("You have provided input data with incoherent sizes")
        for k in range(n_interpolators):
            self.interpolators.append(
                PolynomialInterpolator(x[k:k+2], y[k:k+2],
                                       yd[k:k+2], ydd[k:k+2]))

    def __call__(self, x_q):
        # identify for each entry in x_q the identity of the polynomial to call
        y_q = np.zeros(x_q.shape)
        for interpolator in self.interpolators:
            xq_inds = np.where(np.logical_and(x_q <= interpolator.x[1],
                                              x_q >= interpolator.x[0]))
            y_q[xq_inds] = interpolator.evaluate_polynomial(x_q[xq_inds])
        # handle out of range case
        y_q[x_q < self.x[0]] = self.y[0]
        y_q[x_q > self.x[-1]] = self.y[-1]
        return y_q


def test_polynomial_interpolator_zero_end():
    x = np.array([-0., 3.4])
    y = 10*np.random.randn(2)
    a = PolynomialInterpolator(x, y)
    x_q = np.linspace(-1, 5, 1000)
    plt.plot(x_q, a(x_q))
    plt.scatter(x, y)
    plt.show()

def test_polynomial_interpolator_nonzero_end():
    x = np.array([-0., 3.4])
    y = 10*np.random.randn(2)
    yd = np.array([-1, -1])
    a = PolynomialInterpolator(x, y, yd=yd)
    x_q = np.linspace(-1, 5, 1000)
    plt.plot(x_q, a(x_q))
    plt.scatter(x, y)
    plt.show()

def test_polynomial_interpolator_via_zero_end():
    x = np.array([.0, 1.2,3.6])
    y = np.array([1., 2.4, 1.])
    #yd = np.array([0., 0., 0.])
    ydd = np.array([0.,-2.,0.])
    a = ViaPointInterpolator(x, y,ydd=ydd)
    x_q = np.linspace(-1, 5, 10000)
    plt.plot(x_q, a(x_q))
    plt.scatter(x, y)
    plt.show()

#test_polynomial_interpolator_via_zero_end()
