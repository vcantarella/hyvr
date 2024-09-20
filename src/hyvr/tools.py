import numba
import numpy as np
import scipy
from scipy.spatial.distance import pdist, squareform

from .utils import ferguson_theta_ode, gaussian_kernel, specsim_syn


def ferguson_curve(
    h: np.float64,
    k: np.float64,
    eps_factor: np.float64,
    flow_angle: np.float64,
    s_max: np.float64,
    xstart: np.float64,
    ystart: np.float64,
    extra_noise=0.0,
):
    """
    Simulate extruded parabola centrelines using the Ferguson (1976) disturbed meander model
    Implementation of AR2 autoregressive model
    http://onlinelibrary.wiley.com/doi/10.1002/esp.3290010403/full

    Parameters
    ----------
    h : float
        Height (Ferguson model parameter)
    k : float
        Wave number (Ferguson model parameter)
    eps_factor : float
        Random background noise (Ferguson model parameter)
    flow_angle : float
        Angle of mean flow direction, in radians
    s_max : float
        Length of the domain in mean flow direction
    xstart, ystart : float
        Starting coordinates of the channel centerline
    extra_noise : float
        small error added to the covariance matrix to avoid singular matrix in the underlying Gaussian error curve

    Returns
    -------
    outputs : float matrix
        Simulated extruded parabola centerlines: storage array containing values for x coordinate, y
        coordinate, vx and vy
    """
    # Parameters
    # Calculate curve directions
    theta, s, xp, yp, tau = ferguson_theta_ode(
        s_max, eps_factor, k, h, 0.0, extra_noise
    )  # np.pi / 2)

    # Interpolate curve direction over interval of interest
    # s_interp, th_interp = curve_interp(s, theta, 10)

    # s_diff = np.diff(s)
    # s_diff = np.concatenate([np.array([0.]), s_diff])
    vx = np.cos(theta)
    vy = np.sin(theta)
    # vx = ds*np.cos(th_interp)
    # vy = ds*np.sin(th_interp)
    # xp = np.cumsum(vx)
    # yp = np.cumsum(vy)

    # Storage array
    outputs = np.zeros((len(theta), 5))

    outputs[:, 0] = xp  # x coordinate
    outputs[:, 1] = yp  # y coordinate
    outputs[:, 2] = vx  # vx
    outputs[:, 3] = vy  # vy
    outputs[:, 4] = s

    # Rotate meanders into mean flow direction
    rot_angle = flow_angle  # -np.mean(theta)
    rotMatrix = np.array(
        [
            [np.cos(rot_angle), -np.sin(rot_angle)],
            [np.sin(rot_angle), np.cos(rot_angle)],
        ]
    )
    roro = np.dot(rotMatrix, outputs[:, 0:2].transpose())

    outputs[:, 2:4] = np.dot(rotMatrix, outputs[:, 2:4].transpose()).transpose()
    outputs[:, 0] = roro[0, :].transpose()
    outputs[:, 1] = roro[1, :].transpose()

    # move to channel start
    outputs[:, 0] = outputs[:, 0] + xstart
    outputs[:, 1] = outputs[:, 1] + ystart

    return outputs[:, 0], outputs[:, 1], outputs[:, 2], outputs[:, 3], outputs[:, 4]


def specsim_surface(
    x: np.array,
    y: np.array,
    mean: np.float64,
    var: np.float64,
    corl: np.array,
    mask=None,
):
    """
    Creates gaussian random surface with mean value and variance input
    with the spectral method from  Dietrich & Newsam (1993).
    Parameters:
    -------------
    x,y: 2D grid of x and y points
    mean: mean value
    var: variance
    corl: correlation lenghts (same unit as x and y) in x and y directions
    mask: mask array (same dimensions as x and y)

    Returns
    -------
    Z: output np.array with same dimensions as x and y and with Z values corrensponding to the surface
    """
    M = np.diag(1 / corl ** 2)
    if mask is not None:
        x[~mask] = np.nan
        y[~mask] = np.nan
        coords = [x, y]
        Z = specsim_syn(gaussian_kernel, coords, mean, (var, M))
        Z[~mask] = np.nan
    else:
        coords = [x, y]
        Z = specsim_syn(gaussian_kernel, coords, mean, (var, M))
    return Z


def contact_surface(
    x: np.array,
    y: np.array,
    mean: np.float64,
    var: np.float64,
    corl: np.array,
    mask=None,
):
    """
    Creates gaussian random contact surface with mean value and variance input
    with the spectral method from  Dietrich & Newsam (1993).

    Parameters:
    -------------
    x,y: 2D grid of x and y points
    mean: mean value
    var: variance
    corl: correlation lenghts (same unit as x and y) in x and y directions
    mask: mask array (same dimensions as x and y)

    Returns
    -------
    Z: output np.array with same dimensions as x and y and with Z values corrensponding to the surface
    """
    return None


def surface_gauss_regression(
    x: np.array,
    y: np.array,
    mean: np.float64,
    variance: np.float64,
    corl: np.array,
    dataset: np.array,
    error: np.array,
):
    """
    Performs surface gaussian regression on input x,y data with given dataset (x,y,z) and error
    Based on the algorithm from Rasmussen & Williams (2006) Gaussian Processes for Machine Learning
    Parameters:
    -------------
    x,y: 2D grid of x and y points
    
    Returns
    -------
    Z: output np.array with same dimensions as x and y and with Z values corrensponding to the surface
    """
    # Calculating distances:
    shape = x.shape

    x = np.expand_dims(x.ravel(), axis=1)
    y = np.expand_dims(y.ravel(), axis=1)

    # distance of the training points:
    x_t = np.expand_dims(dataset[:, 0].ravel(), axis=1)
    y_t = np.expand_dims(dataset[:, 1].ravel(), axis=1)
    z_t = dataset[:, 2].ravel()
    x_dist_t = pdist(x_t, "euclidean")
    y_dist_t = pdist(y_t, "euclidean")

    # covariance matrix:
    @numba.njit()
    def kernel_2d(sigma_f, M, x_dist, y_dist) -> np.float64:
        x = np.expand_dims(np.array((x_dist, y_dist)), axis=1)
        x = x.astype(np.float64)
        cov = sigma_f * np.exp(-(1 / 2) * x.T @ M @ x)
        return cov[0, 0]

    M = np.eye(2) * np.array([1 / corl[0] ** 2, 1 / corl[1] ** 2])

    @numba.njit()
    def cov_matrix(sigma_f, x_dist, y_dist, M):
        cov = np.zeros(x_dist.shape[0])
        for i in range(x_dist.shape[0]):
            cov[i] = kernel_2d(sigma_f, M, x_dist[i], y_dist[i])
        return cov

    cov = cov_matrix(variance, x_dist_t, y_dist_t, M)
    cov = squareform(cov)
    cov = cov + np.eye(cov.shape[0])
    error = np.diag(error)
    cov = cov + error
    # Cholesky decomposition
    L, lower = scipy.linalg.cho_factor(cov, lower=True)
    L_y = scipy.linalg.cho_solve((L, lower), z_t)

    # covariance between test and training points:
    @numba.njit()
    def distance(x, y, x_t, y_t):
        x_dist = np.zeros((x.shape[0], x_t.shape[0]))
        y_dist = np.zeros((x.shape[0], x_t.shape[0]))
        for i in range(x.shape[0]):
            x_dist[i, :] = np.ravel((x[i, 0] - x_t) ** 2)
            y_dist[i, :] = np.ravel((y[i, 0] - y_t) ** 2)
        return x_dist, y_dist

    x_dist, y_dist = distance(x, y, x_t, y_t)
    cov_shape = x_dist.shape
    cov_test = cov_matrix(variance, x_dist.ravel(), y_dist.ravel(), M)
    cov_test = cov_test.reshape(cov_shape)
    # Mean prediction
    mean = cov_test.T @ L_y + mean
    # Variance prediction
    # to be implemented v = scipy.linalg.solve_triangular(L, cov_test, lower=True)
    return np.reshape(mean, shape)
