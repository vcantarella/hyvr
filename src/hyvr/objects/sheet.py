import numba
import numpy as np

from ..utils import (
    coterminal_angle,
    get_alternating_facies,
    normal_plane_from_dip_dip_dir,
)


@numba.jit(nopython=True, parallel=False)
def sheet(
    f_array,
    dip_array,
    dip_dir_array,
    x,
    y,
    z,
    xmin,
    xmax,
    ymin,
    ymax,
    bottom_surface,
    top_surface,
    facies,
    internal_layering=False,
    alternating_facies=False,
    dip=0.0,
    dip_dir=0.0,
    layer_dist=0.0,
):
    """
    Assigns a sheet to the grid points x,y,z.
    The sheet is a layer is defined by bounding x and y coordinates and top and bottom contacts.
    It may have internal layering (inclined or planar)
    params:
    ---
    f_array: ndarray(int32) of the facies values at the coordinates (x,y,z)
    dip_array: ndarray(float32) of the dip (positive value) of the internal structure at (x,y,z)
    dip_dir_array: ndarray(float32) of the dip-direction of the internal structure
    x,y,z: grid center coordinates where the search if it is in it or not.
    xmin,xmax: bounding coordinates in the x direction, (normally a sheet should cover the whole domain)
    ymin,ymax: bounding coordinates in the y direction,
    bottom_surface:(float or ndarray): bottom surface of the sheet. It can be a float (planar), or a surface 2D array. In case it is a surface, x,y,z dimensions must match the surface dimensions.
    top_surface (float or ndarray): top surface of the sheet. Defined the same as a bottom_surface.
    facies: np.array(int32) with the facies code (1 in case no layering or more in case of layering)
    internal_layering: True if internal layering
    alternating_facies: True if the facies alternate according to the order in the argument facies
    dip: dip in degrees of the internal dipping layers. Leave the default value for massive structure.
    dip_dir: dip direction in degrees of the internal dipping layers. Leave the default value for massive structure.
    follows the mathematical convention, anticlockwise from east
    layer_dist: perpendicular to dip distance between layers
    
    Modified arrays:
    ---
    f_array: ndarray(int32) of the facies values at the coordinates (x,y,z)
    dip_array: ndarray(float32) of the dip (positive value) of the internal structure at (x,y,z)
    dip_dir_array: ndarray(float32) of the dip-direction of the internal structure
    """
    true_array_x = (x >= xmin) & (x <= xmax)
    true_array_y = (y >= ymin) & (y <= ymax)
    # if len(bottom_surface.shape) != len(z.shape):
    #     bottom_surface = np.broadcast_to(bottom_surface, z.shape)
    # if len(top_surface.shape) != len(z.shape):
    #     top_surface = np.broadcast_to(top_surface, z.shape)

    true_array_z = (z >= np.broadcast_to(bottom_surface, z.shape)) & (
        z <= np.broadcast_to(top_surface, z.shape)
    )
    true_array = true_array_z & true_array_y & true_array_x
    true_array = np.ravel(true_array)
    #facies_output = np.ones(x.size, dtype=np.int32) * (-1)
    if internal_layering:
        normal_vector = normal_plane_from_dip_dip_dir(dip, dip_dir)
        xcenter = xmin + (xmax - xmin) / 2
        ycenter = ymin + (ymax - ymin) / 2
        zmax = np.max(top_surface)
        shift = (
            normal_vector[0] * xcenter
            + normal_vector[1] * ycenter
            + normal_vector[2] * zmax
        )
        d = (
            normal_vector[0] * x.ravel()[true_array]
            + normal_vector[1] * y.ravel()[true_array]
            + normal_vector[2] * z.ravel()[true_array]
            - shift
        )
        class_distances = (np.floor(d / layer_dist)).astype(np.int16)
        min_value = np.min(class_distances)
        facies_indices = class_distances - min_value
        n_layers = int(np.max(facies_indices) + 1)
        facies_array = get_alternating_facies(facies, n_layers, alternating_facies)
        facies_ = np.array([facies_array[n] for n in facies_indices])
        f_array.ravel()[true_array] = facies_
    else:
        f_array.ravel()[true_array] = np.repeat(facies[0], np.sum(true_array))
    dip = np.deg2rad(dip)
    dip_dir = coterminal_angle(dip_dir)
    dip_array.ravel()[true_array] = np.repeat(dip, np.sum(true_array))
    dip_dir_array.ravel()[true_array] = np.repeat(dip_dir, np.sum(true_array))
    #dip = np.repeat(dip, x.size)
    #dip_direction = np.repeat(dip_dir, x.size)
    f_array = np.reshape(f_array, x.shape)
    dip_array = np.reshape(dip_array, x.shape)
    dip_dir_array = np.reshape(dip_dir_array, x.shape)
    # return facies_output, dip, dip_direction
