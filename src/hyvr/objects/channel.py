import numba
import numpy as np

from ..utils import get_alternating_facies, min_distance


@numba.jit(nopython=True, parallel=True)
def channel(
    f_array,
    dip_array,
    dip_dir_array,
    x,
    y,
    z,
    z_top,
    curve,
    parabola_pars,
    facies,
    internal_layering=False,
    alternating_facies=False,
    dip=0.0,
    layer_dist=0.0,
):
    """
    Assigns a channel to the grid points x,y,z.
    The channel is defined by a curve, which represents the
    trajectory of the channel and a parabola, which defines the cross section.
    Besides, it may have internal structure (Not currently implemented).

    params:
    ---
    f_array: ndarray(int32) of the facies values at the coordinates (x,y,z)
    dip_array: ndarray(float32) of the dip (positive value) of the internal structure at (x,y,z)
    dip_dir_array: ndarray(float32) of the dip-direction of the internal structure
    x,y,z: np.arrays of the the center grid points which are to be tested to be assigned to the channel feature.
    z_top: top coordinate (flat) of the channel.
    curve: [x,y,(vx,vy)] 2D array where the first and second columns are the x and y posiiton of the points defining the curve.
    In case the internal layering, then the 3rd and 4th columns must be defined, which are the curve gradient (dc/dx and dc/dy) at each point.
    parabola_pars: [width, depth] of the parabola defining the channel cross section.
    internal_layering: True if internal layering, currently not implemented.
    alternating_facies: True if the facies alternate according to the order in the argument facies
    dip: dip of the internal dipping layers. Leave the default value for massive structure.
    layer_dist: perpendicular to dip distance between layers
    facies: np.array(int32) with the facies code (1 in case no layering or more in case of layering)

    Modified arrays:
    ---
    f_array: ndarray(int32) of the facies values at the coordinates (x,y,z)
    dip_array: ndarray(float32) of the dip (positive value) of the internal structure at (x,y,z)
    dip_dir_array: ndarray(float32) of the dip-direction of the internal structure
    """
    # parabola_args:
    width, depth = parabola_pars

    # if the point is above the channel top, don't consider it
    dz = z - z_top
    dz = dz.ravel()

    # first logic: testing if points are in the z range:
    logic_z = (dz <= 0) & (dz >= -depth)

    # Filter points to the vicinity of the curve:
    xmin, ymin = (
        np.min(curve[:, 0]) - width,
        np.min(curve[:, 1]) - width,
    )
    xmax, ymax = np.max(curve[:, 0]) + width, np.max(curve[:, 1]) + width
    logic_x = (x >= xmin) & (x <= xmax)
    logic_x = logic_x.ravel()
    logic_y = (y >= ymin) & (y <= ymax)
    logic_y = logic_y.ravel()

    # Do heavy calculations only in points that make sense:
    filter_zone = logic_z & logic_x & logic_y
    filter_zone = filter_zone.ravel()
    # Distance in the xy-plane
    P = np.column_stack((x.ravel()[filter_zone], y.ravel()[filter_zone]))
    xy_dist = np.ones(x.size) * 1e10
    idx_curve = np.zeros(x.size, dtype=np.int32)

    # Not a perfect method, to be improved:
    x_curve = curve[:, 0]
    y_curve = curve[:, 1]
    xy_dist[filter_zone], idx_curve[filter_zone] = min_distance(x_curve, y_curve, P)
    logic_xy = np.full(x.size, False)
    logic_xy[filter_zone] = xy_dist[filter_zone] ** 2 <= (
        width**2 / 4 + width**2 * dz[filter_zone] / (4 * depth)
    )  # From the HyVR documentation.

    logic_inside = logic_xy
    if internal_layering:
        # distance between neighbouring points:
        dif = np.diff(np.ascontiguousarray(curve[:, 0:2])) ** 2
        srqt_dif = np.empty_like(dif)
        for i in numba.prange(dif.shape[0]):
            srqt_dif[i] = np.sqrt(np.sum(dif[i]))
        srqt_dif = np.ravel(srqt_dif)
        dist_curve = np.concatenate((np.zeros(1), srqt_dif))
        # gradient values:
        # vx = curve[:, 2]
        # vy = curve[:, 3]
        # azimuth from inverse distance weighted velocity
        # dip and azimuth for the channel internal features is not currently implemented
        # azim = np.where(logic_inside, np.arctan2(vy, vx) / np.pi * 180, -1)
        dip = np.radians(dip)
        # create facies array:
        curve_length = np.sum(dist_curve)
        n_layers = int(
            np.ceil(curve_length * np.sin(dip) / layer_dist)
            + np.ceil(depth / layer_dist)
        )

        n_layers += 10  # just to be sure (why?)
        facies_array = get_alternating_facies(facies, n_layers, alternating_facies)
        # To correct for the distance in z-direction we subtract |dz| * cos(dip)
        # note that dz is negative here
        d = dist_curve * np.sin(dip) + dz * np.cos(dip)
        ns = np.rint(d / layer_dist).astype(np.int32)
        d_grid = np.empty_like(idx_curve)
        ns_grid = np.empty_like(idx_curve)
        for i in numba.prange(idx_curve.shape[0]):
            d_grid[i] = d[idx_curve[i]]
            ns_grid[i] = ns[idx_curve[i]]
        facies = np.array([facies_array[n] for n in ns_grid])
        f_array.ravel()[logic_inside] = facies
    else:
        f_array.ravel()[logic_inside] = np.repeat(facies[0], np.sum(logic_inside))
    dip_array.ravel()[logic_inside] = np.repeat(0.0, np.sum(logic_inside))
    dip_dir_array.ravel()[logic_inside] = np.repeat(0.0, np.sum(logic_inside))
    f_array = np.reshape(f_array, x.shape)
    dip_array = np.reshape(dip_array, x.shape)
    dip_dir_array = np.reshape(dip_dir_array, x.shape)
