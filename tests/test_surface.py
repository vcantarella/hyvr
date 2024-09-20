import numpy as np
from src.hyvr.tools import ferguson_curve, specsim_surface
from src.hyvr.utils import specsim


# Test data
def test_curve():
    h = 0.1
    k = np.pi / 200
    eps_factor = (np.pi / 1.5) ** 2
    channel_curve_1 = ferguson_curve(
        h=h,
        k=k,
        eps_factor=eps_factor,
        flow_angle=0.0,
        s_max=400,
        xstart=0,
        ystart=25,
    )
    x_t, y_t, vx, vy, s = channel_curve_1


def test_specsim():
    x = np.linspace(0, 100, 1000)
    y = np.linspace(0, 100, 1000)
    x, y = np.meshgrid(x, y)
    mean = 2.5
    variance = 0.05**2
    Z = specsim(
        x,
        y,
        mean=mean,
        var=variance,
        corl=np.array([100, 100]),
        mask=None,
        covmod="gaussian",
    )
    assert Z.shape == x.shape
    assert np.allclose(Z.mean(), mean, atol=1e-3)

def test_specsim_surface():
    x = np.linspace(0, 100, 1000)
    y = np.linspace(0, 100, 1000)
    x, y = np.meshgrid(x, y)
    mean = 2.5
    variance = 0.05**2
    corl = np.array([100., 100.])
    Z = specsim_surface(x,y, mean, variance, corl)
    assert Z.shape == x.shape
    assert np.allclose(Z.mean(), mean, atol=1e-3)

