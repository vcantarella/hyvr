import numba
import numpy as np

from hyvr.utils import (
    azimuth_to_counter_clockwise,
    coterminal_angle,
    dip_dip_dir_bulbset,
    get_alternating_facies,
    is_point_inside_ellipsoid,
    normal_plane_from_dip_dip_dir,
)


@numba.jit(nopython=True, parallel=False)
def trough(
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
    # unpacking values:
    x_c, y_c, z_c = center_coords  # coordinates of the center point
    a, b, c = dims  # ellipsoid dimensions
    # auxiliary parameters
    zmin = z_c - c
    zmax = z_c
    # alpha is the azimuth in radians measured from the east axis (x axis)
    alpha = coterminal_angle(azimuth_to_counter_clockwise(azim))
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
    #    return np.ones(original_shape,dtype=np.int64)*-1, np.empty(original_shape,dtype=np.float64), np.empty(original_shape,dtype=np.float64)
    # At this point we know that the point is in the domain, so we have to
    # assign values
    # idx = np.nonzero(gross_limit_logic)[0]
    # dip_dir[idx[logic]] = dip_dir
    # dip[idx[logic]] = dip

    if bulb:
        dip_output, dip_dir_output, norm_distance = dip_dip_dir_bulbset(
            x, y, z, x_c, y_c, z_c, a, b, c, alpha, dip
        )
        dip_output = np.where(logic, dip_output, np.nan)
        dip_dir_output = np.where(logic, dip_dir_output, np.nan)
        if internal_layering:
            n_layers = np.int32(np.ceil(np.max(norm_distance) * c / layer_dist))
            ns = (np.floor((norm_distance) * c / layer_dist)).astype(np.int32)
            facies_array = get_alternating_facies(facies, n_layers, alternating_facies)
            facies_output = np.array([facies_array[n] for n in ns])
            facies_output = np.where(logic, facies_output, -1)
        else:
            facies_output = np.where(logic, facies[0], -1)
    else:  # no bulbset
        if internal_layering:
            normal_vector = normal_plane_from_dip_dip_dir(dip, dip_dir)
            shift = (
                layer_dist
                + normal_vector[0] * x_c
                + normal_vector[1] * y_c
                + normal_vector[2] * z_c
            )
            # X = xmax - xmin
            # Y = ymax - ymin
            # Z = c
            # n_layers = np.max(np.array([X / (np.abs(normal_vector[0]) * layer_dist),
            #                            Y / (np.abs(normal_vector[1]) * layer_dist),
            #                            Z / (np.abs(normal_vector[2]) * layer_dist)]))
            # n_layers = np.int32(n_layers) + 2 + 2

            plane_dist = (
                x * normal_vector[0]
                + y * normal_vector[1]
                + z * normal_vector[2]
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
            facies_output = np.where(logic, facies_output, -1)
        else:
            facies_output = np.where(logic, facies[0], -1)
        dip_output = np.where(logic, dip, np.nan)
        dip_dir_output = np.where(logic, dip_dir, np.nan)

    # reshaping final arrays and assigning values
    facies_final = np.ones(gross_limit_logic.shape, dtype=np.int32) * (-1)
    facies_final[gross_limit_logic] = facies_output
    facies_final = np.reshape(facies_final, original_shape)
    dip_final = np.empty(gross_limit_logic.shape, dtype=np.float64) * np.nan
    dip_final[gross_limit_logic] = dip_output
    dip_final = np.reshape(dip_final, original_shape)
    dip_dir_final = np.empty(gross_limit_logic.shape, dtype=np.float64) * np.nan
    dip_dir_final[gross_limit_logic] = dip_dir_output
    dip_dir_final = np.reshape(dip_dir_final, original_shape)
    return facies_final, dip_final, dip_dir_final
