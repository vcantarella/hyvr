# %%
import numpy as np
from hyvr import trough
from hyvr.tools import ferguson_curve


# %%
def test_trough_angles():
    """
    Tests the angles from the dip and dip direction of a trough in a dense grid.
    """
    # %%creating a grid according to the MODFLOW convention
    xs = np.linspace(-2, 100, 200)
    ys = np.linspace(-2, 80, 160)
    zs = np.linspace(-2, 20, 50)
    z, y, x = np.meshgrid(zs, ys, xs, indexing="ij")
    z = np.flip(z, axis=-2)
    y = np.flip(y, axis=-1)
    centercoord = np.array([48, 40, 19])
    dims = np.array([38, 20, 10])
    # %%
    facies, dip, dip_dir = trough(
        x,
        y,
        z,
        centercoord,
        dims,
        azim=50.0,
        facies=np.array([-1]),
        bulb=True,
        dip=30.0,
    )
    # %%

    assert np.nanmax(dip) == 30.0
    assert np.nanmin(dip) >= 0.0


def test_curve():
    np.random.seed(37)
    channel_curve_1 = ferguson_curve(
        h=0.1,
        k=np.pi / 200,
        eps_factor=(np.pi / 1.5) ** 2,
        flow_angle=0.0,
        s_max=400,
        xstart=40.0,
        ystart=25.0,
    )
    curve = np.column_stack([channel_curve_1[0], channel_curve_1[1]])
    assert curve[0, 0] == 40.0
    assert curve[0, 1] == 25.0
    s = channel_curve_1[4]
    print(s)
    assert s[-1] <= 400.0
