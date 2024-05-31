import numpy as np
from src.hyvr.tools import surface_gauss_regression, ferguson_curve

# Test data
def test_curve():
    h=0.1,
    k=np.pi / 100,
    eps_factor=1,
    channel_curve_1 = ferguson_curve(
    h=0.1,
    k=np.pi / 200,
    eps_factor=(np.pi / 1.5) ** 2,
    flow_angle=0.0,
    s_max=400,
    xstart=0,
    ystart=25,
    )
    x_t, y_t, vx, vy, s = channel_curve_1


def test_gauss_surface_regression():
    
    x = np.linspace(0,100, 1000)
    y = np.linspace(0,100, 1000)
    x, y = np.meshgrid(x, y)
    # training data:
    h=0.4
    k=np.pi / 200
    eps_factor=(np.pi / 1.5) ** 2
    x_t, y_t, vx, vy, s = ferguson_curve(
    h=h,
    k=k,
    eps_factor=eps_factor,
    flow_angle=0.0,
    s_max=400,
    xstart=0,
    ystart=25,
    extra_noise=1e-4,
    )
    z_t = np.sin(s)
    dataset = np.array([x_t, y_t, z_t]).T
    error = np.repeat(0.3, dataset.shape[0])
    corl = np.array([10, 10])
    mean = 0.
    variance = 1.
    #Z = surface_gauss_regression(x, y, mean, variance, corl, dataset, error)
    #assert Z.shape == x.shape
    #assert Z.shape == y.shape
    assert x_t.shape == y_t.shape
    assert x_t.shape == vx.shape
