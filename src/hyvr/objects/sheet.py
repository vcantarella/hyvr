import numpy as np
import numpy.typing as npt
import numba
from hyvr.utils import normal_plane_from_dip_dip_dir
from hyvr.utils import coterminal_angle
from hyvr.utils import get_alternating_facies

@numba.jit(nopython=True, parallel=True)
def sheet(x,y,z,
          xmin,xmax,
          ymin,ymax,
          bottom_surface,top_surface,
          facies,
          internal_layering=False, alternating_facies=False,
          dip=0.,dip_dir=0.,layer_dist=0.
          ):
    """
    Assigns a sheet to the grid points x,y,z.
    The sheet is a layer is defined by bounding x and y coordinates and top and bottom contacts.
    It may have internal layering (inclined or planar)
    params:
    ---
    x,y,z: grid coordinates where the element will search for connection.
    xmin,xmax: bounding coordinates in the x direction, (normally a sheet should cover the whole domain)
    ymin,ymax: bounding coordinates in the y direction,
    bottom_surface:(float or ndarray): bottom surface of the sheet. It can be a float (planar), or a surface 2D array. In case it is a surface, x,y,z dimensions must match the surface dimensions.
    top_surface (float or ndarray): top surface of the sheet. Defined the same as a bottom_surface.
    internal_layering: True if internal layering
    alternating_facies: True if the facies alternate according to the order in the argument facies
    dip: dip of the internal dipping layers. Leave the default value for massive structure.
    layer_dist: perpendicular to dip distance between layers
    facies: np.array(int32) with the facies code (1 in case no layering or more in case of layering)
    
    Returns:
    ---
    facies: ndarray(int32) of the facies values at the coordinates (x,y,z)
    dip: ndarray(float32) of the dip (positive value) of the internal structure at (x,y,z)
    dip_direction: ndarray(float32) of the dip-direction of the internal structure
    """
    true_array_x = (x >= xmin) & (x <= xmax)
    print(true_array_x.shape)
    true_array_y = (y >= ymin) & (y <= ymax)
    # if len(bottom_surface.shape) != len(z.shape):
    #     bottom_surface = np.broadcast_to(bottom_surface, z.shape)
    # if len(top_surface.shape) != len(z.shape):
    #     top_surface = np.broadcast_to(top_surface, z.shape)

    true_array_z = (z >= np.broadcast_to(bottom_surface,z.shape)) & (z<= np.broadcast_to(top_surface,z.shape))
    print(true_array_z.shape)
    true_array = true_array_z & true_array_y & true_array_x
    true_array = np.ravel(true_array)
    print(true_array.shape)
    facies_output = np.ones(x.size, dtype = np.int32)*(-1)
    if internal_layering:
        normal_vector = normal_plane_from_dip_dip_dir(dip, dip_dir)
        xcenter = xmin + (xmax -xmin)/2
        ycenter = ymin + (ymax - ymin)/2
        zmax = np.max(top_surface)
        shift = normal_vector[0] * xcenter + normal_vector[1] * ycenter + normal_vector[2] * zmax
        d = normal_vector[0] * x.ravel()[true_array] + normal_vector[1] * y.ravel()[true_array] + normal_vector[2] * z.ravel()[true_array] - shift
        print(d.shape)
        class_distances = (np.floor(d/layer_dist)).astype(np.int16)
        min_value = np.min(class_distances)
        facies_indices = class_distances - min_value
        n_layers = int(np.max(facies_indices)+1)  
        print(n_layers)
        facies_array = get_alternating_facies(facies, n_layers, alternating_facies)
        facies_ = np.array([facies_array[n] for n in facies_indices])
        facies_output[true_array] = facies_
    else:
        facies_output[true_array] = np.repeat(facies[0], np.sum(true_array))
    dip = np.deg2rad(dip)
    dip_dir = coterminal_angle(np.deg2rad(dip_dir))
    dip = np.repeat(dip, x.size)
    dip_direction = np.repeat(dip_dir, x.size)
    facies_output = np.reshape(facies_output, x.shape)
    dip = np.reshape(dip, x.shape)
    dip_direction = np.reshape(dip_direction, x.shape)
    return facies_output, dip, dip_direction