import numba
import numpy as np

from ..utils import (
    coterminal_angle,
    dip_dip_dir_bulbset,
    get_alternating_facies,
    is_point_inside_ellipsoid,
    normal_plane_from_dip_dip_dir,
)


@numba.jit(nopython=True, parallel=False)
def half_ellipsoid(
    f_array,
    dip_array,
    dip_dir_array,
    x,
    y,
    z,
    center_coords,
    dims,
    azim,
    facies,
    internal_layering=False,
    alternating_facies=False,
    bulb=False,
    dip=0.0,
    dip_dir=0.0,
    layer_dist=0.0,
):
    """
    Assigns a half ellipsoid (trough) to the grid points x,y,z.
    Half ellipsoid is the lower half of an ellipsoid, defined by its center, dimensions and azimuth.
    It can be used to model discrete lenses, trough structure, scour pool fills, etc.

    params:
    ---
    f_array: ndarray(int32) of the facies values at the coordinates (x,y,z)
    dip_array: ndarray(float32) of the dip (positive value) of the internal structure at (x,y,z)
    dip_dir_array: ndarray(float32) of the dip-direction of the internal structure
    x,y,z: grid center coordinates where the search if it is in it or not.
    center_coords: tuple with the x,y,z coordinates of the center of the ellipsoid
    dims: tuple with the dimensions of the ellipsoid (a,b,c)
    azim: azimuth in degrees of the major axis of the ellipsoid. It is measured from the east axis (x axis).
     It follows the mathematical convention, anticlockwise from east.
    facies: np.array(int32) with the facies code (1 in case no layering or more in case of layering)
    internal_layering: True if internal layering
    alternating_facies: True if the facies alternate according to the order in the argument facies
    bulb: True if the ellipsoid is a bulbset. In this case, the internal structure is calculated from the distance to the center.
    dip: dip in degrees of the internal dipping layers. Leave the default value for massive structure.
    dip_dir: dip direction in degreesof the internal dipping layers. Leave the default value for massive structure.
    follows the mathematical convention, anticlockwise from east
    layer_dist: perpendicular to dip distance between layers

    Modified arrays:
    ---
    f_array: ndarray(int32) of the facies values at the coordinates (x,y,z)
    dip_array: ndarray(float32) of the dip (positive value) of the internal structure at (x,y,z)
    dip_dir_array: ndarray(float32) of the dip-direction of the internal structure
    """

    # unpacking values:
    x_c, y_c, z_c = center_coords  # coordinates of the center point
    a, b, c = dims  # ellipsoid dimensions
    # auxiliary parameters
    zmin = z_c - c
    zmax = z_c
    # alpha is the azimuth in radians measured from the east axis (x axis)
    alpha = coterminal_angle(azim)
    # it is faster to calculate a gross limit and later refine the calculations:
    sin_alpha = np.sin(alpha)
    cos_alpha = np.cos(alpha)
    xmin = x_c - np.sqrt(a**2 * cos_alpha**2 + b**2 * sin_alpha**2)
    xmax = x_c + np.sqrt(a**2 * cos_alpha**2 + b**2 * sin_alpha**2)
    ymin = y_c - np.sqrt(a**2 * sin_alpha**2 + b**2 * cos_alpha**2)
    ymax = y_c + np.sqrt(a**2 * sin_alpha**2 + b**2 * cos_alpha**2)
    gross_limit_logic = (
        (x >= xmin)
        & (x <= xmax)
        & (y >= ymin)
        & (y <= ymax)
        & (z >= zmin)
        & (z <= zmax)
    )

    # copy for later
    original_shape = x.shape
    # modifying the reference so we calculate on less points:
    gross_limit_logic = np.ravel(gross_limit_logic)
    x = x.ravel()[gross_limit_logic]
    y = y.ravel()[gross_limit_logic]
    z = z.ravel()[gross_limit_logic]

    # gross_limit_logic = np.ravel(gross_limit_logic)
    # # To decide whether the point is inside, we can just use the normalized
    # # distance. Therefore, we first calculate the normalized distance vector

    # l2 = normalized_dx ** 2 + normalized_dy ** 2 + normalized_dz ** 2
    # logic = l2 < 1
    logic = is_point_inside_ellipsoid(x, y, z, x_c, y_c, z_c, a, b, c, alpha)
    # # To ensure the semi-ellipse shape, we cut the ellipse in half by assuming a negative distance:
    dz = z - z_c
    logic = logic & (np.ravel(dz) <= 0)

    # print(np.sum(logic))
    if np.sum(logic) == 0:  # return empty dataset:
        print("No points inside the ellipsoid")
        return
    
    x_e = x[logic]
    y_e = y[logic]
    z_e = z[logic]

    if bulb:
        dip_output, dip_dir_output, norm_distance = dip_dip_dir_bulbset(
            x_e, y_e, z_e, x_c, y_c, z_c, a, b, c, alpha, dip
        )
        #dip_output = np.where(logic, dip_output, np.nan)
        #dip_dir_output = np.where(logic, dip_dir_output, np.nan)
        if internal_layering:
            n_layers = np.int32(np.ceil(np.max(norm_distance) * c / layer_dist))
            ns = (np.floor((norm_distance) * c / layer_dist)).astype(np.int32)
            facies_array = get_alternating_facies(facies, n_layers, alternating_facies)
            facies_output = np.array([facies_array[n] for n in ns])
            
        else:
            facies_output = np.repeat(facies, np.sum(logic))
    else:  # no bulbset
        if internal_layering:
            normal_vector = normal_plane_from_dip_dip_dir(dip, dip_dir)
            shift = (
                layer_dist
                + normal_vector[0] * x_c
                + normal_vector[1] * y_c
                + normal_vector[2] * z_c
            )
            plane_dist = (
                x_e * normal_vector[0]
                + y_e * normal_vector[1]
                + z_e * normal_vector[2]
                - shift
            )
            plane_dist = np.ravel(plane_dist)
            n_layers = np.ceil(np.max(np.abs(plane_dist) / layer_dist)) * 2
            n_layers = int(n_layers)
            facies_array = get_alternating_facies(facies, n_layers, alternating_facies)
            # get index in facies vector, where n = len(facies)//2 at d=0
            ns = np.floor(plane_dist / layer_dist) + (n_layers // 2)
            ns = ns.astype(np.int32)
            facies_output = np.array([facies_array[n] for n in ns])
            
        else:
            facies_output = np.repeat(facies, np.sum(logic))
        dip = np.deg2rad(dip)
        dip_dir = coterminal_angle(dip_dir)
        dip_output = np.repeat(dip, np.sum(logic))
        dip_dir_output = np.repeat(dip_dir, np.sum(logic))

    # reshaping final arrays and assigning values
    assignment_f = np.zeros(np.sum(gross_limit_logic), dtype=np.int32)
    assignment_f[logic] = facies_output
    assignment_f[~logic] = f_array.ravel()[gross_limit_logic][~logic]
    f_array.ravel()[gross_limit_logic] = assignment_f
    assignment_dip = np.zeros(np.sum(gross_limit_logic), dtype=np.float32)
    assignment_dip[logic] = dip_output
    assignment_dip[~logic] = dip_array.ravel()[gross_limit_logic][~logic]
    dip_array.ravel()[gross_limit_logic] = assignment_dip
    assignment_dip_dir = np.zeros(np.sum(gross_limit_logic), dtype=np.float32)
    assignment_dip_dir[logic] = dip_dir_output
    assignment_dip_dir[~logic] = dip_dir_array.ravel()[gross_limit_logic][~logic]
    dip_dir_array.ravel()[gross_limit_logic] = assignment_dip_dir
    #Wdip_dir_array.reshape(original_shape)
    #return facies_final, dip_final, dip_dir_final
