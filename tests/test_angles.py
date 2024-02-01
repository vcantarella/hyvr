# %%
import numpy as np
from hyvr.objects.trough import trough


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
