import numpy as np
from hyvyr.tools import surface_gauss_regression, ferguson_curve

# Test data
def test_gauss_surface_regression():
    x = np.linspace(0,100, 1000)
    y = np.linspace(0,100, 1000)
    x, y = np.meshgrid(x, y)
    # training data:
    h=0.1,
    k=np.pi / 200,
    eps_factor=(np.pi / 1.5) ** 2,
    x_t, y_t, vx, vy, s = ferguson_curve(h, k, eps_factor, 0, 200, 50, 50)
    z_t = np.sin(s)
    dataset = np.array([x_t, y_t, z_t]).T
    error = np.random.normal(0, 0.1, len(x_t))
    corl = np.array([10, 10])
    mean = 0
    variance = 1
    Z = surface_gauss_regression(x, y, mean, variance, corl, dataset, error)
    assert Z.shape == x.shape
    assert Z.shape == y.shape
