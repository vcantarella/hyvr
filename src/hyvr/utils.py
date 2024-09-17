from typing import List

import numba
import numpy as np
import numpy.typing as npt
import scipy



@numba.njit()
def normal_plane_from_dip_dip_dir(dip: float, dip_dir: float) -> npt.ArrayLike:
    """
    Calculates the normal plane from dip/dip_dir information
    saves the coordinates (X,Y,Z) of the normal vector in a numpy-array.
    Parameters
    ----------
    dip : float
        dip: mathematical convention, anticlockwise from east, in degrees
    dip_dir : float
        dip direction:  mathematical convention, anticlockwise from east, in degrees
    Returns
    -------
    normal vector of the plane (x,y,z): ndarray
    """
    dip = np.deg2rad(dip)
    dip_dir = np.deg2rad(dip_dir + 90)
    normal_vec = np.zeros(3)
    normal_vec[0] = -np.sin(dip) * np.cos(dip_dir)
    normal_vec[1] = np.sin(dip) * np.sin(dip_dir)
    normal_vec[2] = np.cos(dip)
    return normal_vec


@numba.njit()
def rotation_matrix_x(alpha):
    arr = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(alpha), -np.sin(alpha)],
            [0.0, np.sin(alpha), np.cos(alpha)],
        ]
    )
    return arr


@numba.njit()
def rotation_matrix_z(alpha):
    arr = np.array(
        [
            [np.cos(alpha), -np.sin(alpha), 0.0],
            [np.sin(alpha), np.cos(alpha), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return arr


@numba.njit()
def is_point_inside_ellipsoid(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    x_center: float,
    y_center: float,
    z_center: float,
    a: float,
    b: float,
    c: float,
    alpha: float,
) -> np.ndarray:
    """
    Return boolean array with true indices corresponding to points inside the ellipsoid
    Parameters
    ----------
    x,y,z: point coordinates array
    x_center,y_center,z_center: ellipsoid center
    a,b,c: major, minor and vertical axis of the ellipsoid
    alpha: azimuth in counterclockwise from east and in radians
    """
    matrix = rotation_matrix_z(alpha)
    x = np.ravel(x)
    y = np.ravel(y)
    z = np.ravel(z)
    dx = x - x_center
    dy = y - y_center
    dz = z - z_center
    rotated_points = matrix @ np.vstack((dx, dy, dz))
    distance_vector = (
        rotated_points[0, :] ** 2 / a**2
        + rotated_points[1, :] ** 2 / b**2
        + rotated_points[2, :] ** 2 / c**2
    )
    logic = np.where(distance_vector <= 1, True, False)
    logic = np.reshape(logic, x.shape)
    return logic


@numba.njit()
def dip_dip_dir_bulbset(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    x_c: float,
    y_c: float,
    z_c: float,
    a: float,
    b: float,
    c: float,
    alpha: float,
    dip: float,
):
    """
    Return dip and dip_direction_arrays of the points inside ellipsoid according to the bulb method
    Parameters
    ----------
    x,y,z: point coordinates array
    x_c,y_c,z_c: ellipsoid center
    a,b,c: major, minor and vertical axis of the ellipsoid
    alpha: azimuth in counterclockwise from east and in radians
    """
    # Since ellipsoids are isosurfaces of quadratic functions, we can
    # get the normal vector by taking the gradient of the quadratic
    # function that has our ellipsoid as iso-surface.
    # This function can be written as:
    #
    # f(d(x)) = (x*cos+y*sin)**2/a**2 + (-x*sin+y*cos)**2/b** + z**2/c**2
    #
    # The gradient is (up to a scalar)
    #
    #             /  nx*cos/a + ny*sin/b  \
    # grad f(x) = | -nx*sin/a + ny*cos/b  |
    #             \          nz/c         /
    #
    # where nx, ny, nz are the normalized distances.
    # The gradient points outwards.
    # The length of the vector is
    #
    # |grad f(x)| = (nx/a)**2 + (ny/b)**2 + (nz/c)**2
    #
    # The dip angle is the the angle between the normal
    # vector and the unit z-vector.
    # The azimuth is the angle between the projection of the normal
    # vector onto the x-y-plane and the unit x-vector.
    dx = x - x_c
    dx = np.ravel(dx)
    dy = y - y_c
    dy = np.ravel(dy)
    dz = z - z_c
    dz = np.ravel(dz)
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)
    normalized_dx = (dx * cos_alpha + dy * sin_alpha) / a
    normalized_dy = (-dx * sin_alpha + dy * cos_alpha) / b
    normalized_dz = dz / c
    aaa = normalized_dx / a
    bbb = normalized_dy / b
    ccc = normalized_dz / c
    len_normvec = np.sqrt(aaa**2 + bbb**2 + ccc**2)
    normvec_x = (aaa * cos_alpha + bbb * sin_alpha) / len_normvec
    normvec_y = (-aaa * sin_alpha + bbb * cos_alpha) / len_normvec
    normvec_z = ccc / len_normvec

    len_normvec_xy = np.sqrt(normvec_x**2 + normvec_y**2)
    norm_distance = np.sqrt(normalized_dx**2 + normalized_dy**2 + normalized_dz**2)

    # The dip angle can be found as acos of the normalized
    # scalar product of unit z-vector and normal vector
    # It should be negative, if the gradient points in positive
    # x direction and in positive z direction, or if it points
    # in negative x and z direction.
    # direction.
    dip_bulb = (
        -np.arccos(np.abs(normvec_z)) * np.sign(normvec_x * normvec_z) / np.pi * 180
    )

    # The azimuth angle should also be between -90 and 90. It is
    # the angle between the projection of the normal vector into
    # the xy-plane and the x unit vector.
    # It can be calculated as negative atan of the y component
    # over the x component. To make sure the angle is between
    # -90 and 90, we change the sign of both components, if the
    # x component is negative. This means the x component is
    # always positive, and the y component is multiplied by the
    # sign of the x component
    dip_dir_counterclockwise = (
        -np.arctan2(np.sign(normvec_x) * normvec_y, np.abs(normvec_x)) / np.pi * 180
    )
    dip_dir_bulb = (-dip_dir_counterclockwise + 90) % 360

    # if len_normvec != 0:
    # the point is exactly at the center of the trough, this means
    # there is no azim and dip
    dip_dir_output = np.where(len_normvec < 1e-12, 0.0, dip_dir_bulb)
    dip_output = np.where(len_normvec < 1e-12, 0.0, dip_bulb)

    # normal vector points in z-direction -> dip = 0, azim = 0
    dip_dir_output = np.where(len_normvec_xy < 1e-12, 0.0, dip_dir_output)
    dip_output = np.where(len_normvec_xy < 1e-12, 0.0, dip_output)

    # for geological applications, a negative dip does not make sense:
    dip_dir_output = np.where(
        dip_output < 0,
        np.where(dip_dir_output < 180, dip_dir_output + 180, dip_dir_output - 180),
        dip_dir_output,
    )
    dip_output = np.abs(dip_output)
    # cap dip in bulb or bulbsets: use dip as maximum dip
    dip_output = np.where(dip_output > dip, dip, dip_output)
    dip_dir_output = np.deg2rad(dip_dir_output)
    dip_output = np.deg2rad(dip_output)

    return dip_output, dip_dir_output, norm_distance


@numba.njit()
def get_alternating_facies(facies, n_layers, facies_ordering) -> np.ndarray:
    """
    Returns a vector of alternating facies numbers
    """
    if facies_ordering:
        facies_array = alternating_facies(facies, n_layers)
    else:
        facies_array = np.random.choice(facies, n_layers)
    return facies_array


@numba.njit()
def alternating_facies(facies, n_layers):
    initial_size = np.int32(facies.size)
    reps = np.int32(np.ceil(n_layers / facies.size))
    facies_array = np.repeat(facies, reps)
    facies_array = np.reshape(facies_array, (initial_size, reps))
    final_array = facies_array.transpose().copy()
    final_array = np.reshape(final_array, final_array.size)
    final_array = final_array[:n_layers]
    return final_array


@numba.njit()
def coterminal_angle(angle):
    "Ensure the angle is in the range [0, 360) degrees"
    # Ensure the angle is in the range [0, 360) degrees
    normalized_angle = angle % 360
    # Ensure the angle is in the range [0, 2π) radians
    normalized_angle_radians = np.deg2rad(normalized_angle)
    return normalized_angle_radians


@numba.njit()
def azimuth_to_counter_clockwise(azimuth: npt.ArrayLike) -> npt.ArrayLike:
    "Converts azimuth angle to counter clockwise angle in degrees"
    h = 450 - azimuth
    degree = np.where(h >= 360, h - 360, h)
    return degree


@numba.njit()
def sign(x: float):
    return (x >= 0) - (x < 0)


@numba.njit(nogil=True)
def distance(X, x, y):
    return np.sqrt((X[0] - x) ** 2 + (X[1] - y) ** 2)


@numba.jit(nopython=True, parallel=True)
def min_distance(x, y, P):
    """
    Compute minimum/a distance/s between
    a point P[x0,y0] and a curve (x,y)

    ARGS:
        x, y      (array of values in the curve)
        P         (array of points to sample)

    Returns min indexes and distances array.
    """
    # compute distance
    d_array = np.zeros((P.shape[0]))
    glob_min_idx = np.zeros((P.shape[0]))
    for i in numba.prange(P.shape[0]):
        d_line = distance(P[i], x, y)
        d_array[i] = np.min(d_line)
        glob_min_idx[i] = np.argmin(d_line)
    return d_array, glob_min_idx


def ferguson_theta_ode(
    s_max: np.float64,
    eps_factor: np.float64,
    k: np.float64,
    h: np.float64,
    omega: np.float64,
    err: np.float64 = 1e-8,
):
    """
    Implementation of  (Ferguson, 1976, Eq.9).
    The equation is formulated as an initial value problem and integrated with scipy function for integration (solve_ivp)
    http://onlinelibrary.wiley.com/doi/10.1002/esp.3290010403/full

    Parameters:
        s_max: 	max length of channel
        k:		Wavenumber
        h:		Height
        eps_factor:	Random background noise (normal variance)
        omega: initial angle
        err: small error added to the covariance matrix to avoid singular matrix

    Returns:
        theta : angle array
        s : channel distance array

    """

    # Correlated variance calculation => Gaussian function
    s_range = np.arange(0, s_max, (s_max / 1000))
    dist_arr = scipy.spatial.distance.pdist(
        np.expand_dims(s_range, axis=1), "sqeuclidean"
    )
    variance = eps_factor + err
    cov = variance * np.exp(-(1 / 2) * dist_arr)

    cov = scipy.spatial.distance.squareform(cov)
    cov[np.diag_indices_from(cov)] = variance
    u = np.random.multivariate_normal(
        np.zeros_like(s_range), np.eye((s_range).shape[0])
    )
    cov = cov + err * np.eye(cov.shape[0])
    L = scipy.linalg.cholesky(cov)
    e_s = L @ u

    def rhs(t, y, k, h):
        eps_t = np.interp(np.array([t]), s_range, e_s)
        eps_t = eps_t[0]  # from array to float
        d_tau_ds = (eps_t - y[1] - 2 * h / k * y[0]) * (k**2)
        d_theta_ds = y[0]
        dx_ds = np.cos(y[1])
        dy_ds = np.sin(y[1])
        return np.array([d_tau_ds, d_theta_ds, dx_ds, dy_ds])

    def jac(t, y, k, h):
        return np.array(
            [
                [-2 * h * k, -(k**2), 0, 0],
                [1, 0, 0, 0],
                [0, -np.sin(y[1]), 0, 0],
                [0, np.cos(y[1]), 0, 0],
            ]
        )

    y0 = np.array([omega * k, 0.0, 0.0, 0.0])

    solution = scipy.integrate.solve_ivp(
        rhs,
        (0, s_max),
        y0,
        method="BDF",
        args=(k, h),
        first_step=0.01,
        jac=jac,
        atol=1e-8,
        rtol=1e-8,
    )

    y = solution.y

    s = solution.t

    tau = y[0, :]
    theta = y[1, :]
    x = y[2, :]
    y = y[3, :]

    return theta, s, x, y, tau


def R_1(s, s_arr, curv_arr, k_1, Cf, W, D, Omega=-1, F=2.5):
    # interpolate to find the value of tau at s
    tau = np.interp(s, s_arr, curv_arr)
    Ro = k_1 * tau * W
    s_prime = np.where(s_arr < s, s_arr, 0)
    s_prime = s_prime[s_prime != 0]
    curv_prime = curv_arr[: len(s_prime)]
    sau = s - s_prime
    G_sau = np.exp(-2 * Cf * sau / D)
    Ro_prime = k_1 * curv_prime * W
    integration = np.trapz(Ro_prime * G_sau, sau) / np.trapz(G_sau, sau)
    return Omega * Ro + F * integration


def Rs(s_arr, curv_arr, k_1, W, Cf, D, Omega=-1, F=2.5):
    Rs_arr = np.zeros(len(s_arr))
    for i in range(len(s_arr)):
        Rs_arr[i] = R_1(s_arr[i], s_arr, curv_arr, k_1, Cf, W, D, Omega, F)
    return Rs_arr


def howard_knudson_ode(
    s_max,
    eps_factor,
    k,
    h,
    omega,
    k_1,
    Cf,
    Omega=-1,
    F=2.5,
):
    """
    Implementation of  (Ferguson, 1976, Eq.9).
    The equation is formulated as an initial value problem and integrated with scipy function for integration (solve_ivp)
    http://onlinelibrary.wiley.com/doi/10.1002/esp.3290010403/full

    Parameters:
        s_max: 	max length of channel
        k:		Wavenumber
        h:		Height
        eps_factor:	Random background noise (normal variance)
        omega: initial angle

    Returns:
        theta : angle array
        s : channel distance array

    """

    # Correlated variance calculation => Gaussian function
    s_range = np.arange(0, s_max, (s_max / 1000))
    dist_arr = scipy.spatial.distance.pdist(
        np.expand_dims(s_range, axis=1), "sqeuclidean"
    )
    variance = eps_factor
    cov = variance * np.exp(-(1 / 2) * dist_arr)

    cov = scipy.spatial.distance.squareform(cov)
    cov[np.diag_indices_from(cov)] = variance
    u = np.random.multivariate_normal(
        np.zeros_like(s_range), np.eye((s_range).shape[0])
    )
    L = scipy.linalg.cholesky(cov)
    e_s = L @ u

    def rhs(t, y, k, h):
        eps_t = np.interp(np.array([t]), s_range, e_s)
        eps_t = eps_t[0]  # from array to float
        d_tau_ds = (eps_t - y[1] - 2 * h / k * y[0]) * (k**2)
        d_theta_ds = y[0]
        dx_ds = np.cos(y[1])
        dy_ds = np.sin(y[1])
        return np.array([d_tau_ds, d_theta_ds, dx_ds, dy_ds])

    def jac(t, y, k, h):
        return np.array(
            [
                [-2 * h * k, -(k**2), 0, 0],
                [1, 0, 0, 0],
                [0, -np.sin(y[1]), 0, 0],
                [0, np.cos(y[1]), 0, 0],
            ]
        )

    y0 = np.array([omega * k, 0.0, 0.0, 0.0])

    solution = scipy.integrate.solve_ivp(
        rhs,
        (0, s_max),
        y0,
        method="BDF",
        args=(k, h),
        first_step=0.01,
        jac=jac,
        atol=1e-8,
        rtol=1e-8,
    )

    y = solution.y

    s = solution.t

    theta = y[1, :]
    x = y[2, :]
    y = y[3, :]

    return theta, s, x, y


def gaussian_kernel(r, phi, M):
    """
    Gaussian kernel function

    Parameters
    ----------
    r : np.ndarray
        Distance vector between two points
    phi : float
        Variance of the gaussian kernel
    M : np.ndarray
        Matrix with the correlation structure of the kernel
    """
    r = np.atleast_2d(r)
    # Perform the vector-matrix multiplication with einsum (r.T @ M @ r, for stacked vectors in r)
    result =  phi * np.exp(-0.5 * np.einsum('bi,ij,bj->b', r, M, r))
    
    return result



def matern_kernel(r, nu, M):
    """
    Matern kernel function

    Parameters
    ----------
    r : np.ndarray
        Distance vector between two points
    nu : float
        order of the matern kernel
    M : np.ndarray
        Matrix with the correlation structure of the kernel

    """
    r = np.atleast_2d(r)
    r2 = np.einsum('bi,ij,bj->b', r, M, r)
    kv_ = scipy.special.kv(nu, np.sqrt(2 * nu * r2))
    gamma_ = (
        2 ** (1 - nu)
        / scipy.special.gamma(nu)
        * (np.sqrt(2 * nu * r2)) ** nu
    )
    return gamma_ * kv_


def specsim_syn(kernel, coords: List[np.ndarray], mean=0.0, args=()):
    """
    Generate random variables with stationary covariance function using spectral
    techniques of Dietrich & Newsam (1993)

    Parameters
    ----------
    coords: (x,y,z,...) : 1D, 2D or 3D np.ndarray with x,y,z, coordinates of the grid. Dimensions must match.
    mean: mean value of the random field
    kernel: callable
        Function which computes the covariance function between two points.
        the of the function must be the distance vector between the points
    args: tuple
        Contain the aditional parameters of the covariance function
         (eg. variance, correlation structure, etc. Empty by default.
        The calling signature is ``kernel(x, *args)

    Returns
    -------
    Y : 1d, 2d, or 3d numpy array
        Numpy array of random field given in the same dimensions of x, y and z.
    """
    # calculate the distance to centre
    shape = coords[0].shape
    for i in range(len(coords)):
        coords[i] = coords[i].ravel() - np.nanmean(coords[i])
    # calculate the kernel distance:
    r = np.stack(coords)
    r = r.T
    ryy = kernel(r, *args)
    ryy = ryy.reshape(shape)
    # FFT of the kernel and calculations according to Dietrich & Newsam (1993)
    ntot = ryy.size
    syy = np.fft.fftn(np.fft.fftshift(ryy)) / ntot
    syy = np.abs(syy)  # Remove imaginary artifacts
    syy[0] = 0
    real = np.random.randn(*syy.shape)
    imag = np.random.randn(*syy.shape)
    epsilon = real + 1j * imag
    rand = epsilon * np.sqrt(syy)
    Y = np.real(np.fft.ifftn(rand * ntot))
    Y = Y + mean
    return Y


# so far this function is not jitted. I plan to adapt a gaussian random function generator
# using linear algebra, instead of fft.The linalg module is accessible in numba.
# still buggy!!
def specsim(
    x: np.array,
    y: np.array,
    z=np.array([0.0]),
    mean=0.0,
    var=1.0,
    corl=np.array([1.0, 1.0]),
    z_axis=0,
    mask=None,
    covmod="gaussian",
):
    """
    Generate random variables with stationary covariance function using spectral
    techniques of Dietrich & Newsam (1993)

    Parameters
    ----------
    x,y,z : 3D np.ndarray with x,y,z, coordinates of the grid.
    mean: mean surface z (or 3d variable)
    var : float
        variance of the gaussian value.
    corl : tuple of floats
        Tuple of correlation lengths. 2-tuple for 2-d (x and y), and 3-tuple
        for 3-d (x, y, z).
    z_axis: int
        location of the z_axis in the numpy grid.
        The refernce value is axis 0 (the first), as in the MODFLOW convention.
    mask: np.array optional:
        adds a mask to calculate the gaussian random field in a subset of the grid: x[mask]
    covmod : str, optional (default: "gaussian")
        Which covariance model to use ("gaussian" or "exp").

    Returns
    -------
    Y : 1d, 2d, or 3d numpy array
        Numpy array of random field. If no selection mask was given, this is either a 2d or 3d numpy
        array, depending on ``two_dim`` and of the size of the model grid.
        If a selection_mask was given, this is a flat array of values.
    """
    if covmod not in ["gaussian", "exp"]:
        raise ValueError("covariance model must be 'gaussian' or 'exp'")
    if mask is None:
        x_calc = x - np.mean(x)
        y_calc = y - np.mean(y)
    else:
        x_calc = np.where(mask, x, np.nan)
        x_calc = x_calc - np.nanmean(x_calc)
        y_calc = np.where(mask, y, np.nan)
        y_calc = y_calc - np.nanmean(y_calc)
    two_dim = len(corl) < 3  # boolean weather calculations should be done in two or 3D
    if two_dim:
        Y = np.empty(x.shape)
        h_square = 0.5 * (x_calc / corl[0]) ** 2 + 0.5 * (y_calc / corl[1]) ** 2
    else:
        if mask is None:
            z_calc = z - np.mean(z)
        else:
            z_calc = np.where(mask, z, z_calc)
            z_calc = z_calc - np.nanmean(z_calc)
        Y = np.empty(z.shape)
        h_square = 0.5 * (
            (x_calc / corl[0]) ** 2 + (y_calc / corl[1]) ** 2 + (z_calc / corl[2]) ** 2
        )
    ntot = h_square.size
    # Covariance matrix of variables
    if covmod == "gaussian":
        # Gaussian covariance model
        ryy = np.exp(-h_square) * var
    else:  # covmod == 'exp':
        # Exponential covariance model
        ryy = np.exp(-np.sqrt(h_square)) * var
    # Power spectrum of variable
    syy = np.fft.fftn(np.fft.fftshift(ryy)) / ntot
    syy = np.abs(syy)  # Remove imaginary artifacts
    syy[0] = 0
    real = np.random.randn(*syy.shape)
    imag = np.random.randn(*syy.shape)
    epsilon = real + 1j * imag
    rand = epsilon * np.sqrt(syy)
    Y = np.real(np.fft.ifftn(rand * ntot))
    print(Y.shape)
    Y = Y + mean
    return Y
