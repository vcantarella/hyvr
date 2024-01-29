import numpy as np
from utils import ferguson_theta_ode,specsim

def ferguson_curve(h, k, eps_factor, flow_angle, s_max, xstart, ystart):
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

    Returns
    -------
    outputs : float matrix
        Simulated extruded parabola centerlines: storage array containing values for x coordinate, y
        coordinate, vx and vy
    """
    # Parameters
    # Calculate curve directions
    theta, s, xp, yp = ferguson_theta_ode(s_max, eps_factor, k, h, 0.)#np.pi / 2)

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
    rot_angle = flow_angle #-np.mean(theta)
    rotMatrix = np.array([[np.cos(rot_angle), -np.sin(rot_angle)],
                          [np.sin(rot_angle), np.cos(rot_angle)]])
    roro = np.dot(rotMatrix, outputs[:, 0:2].transpose())

    outputs[:, 2:4] = np.dot(rotMatrix, outputs[:, 2:4].transpose()).transpose()
    outputs[:, 0] = roro[0, :].transpose()
    outputs[:, 1] = roro[1, :].transpose()

    # move to channel start
    outputs[:, 0] = outputs[:, 0] + xstart
    outputs[:, 1] = outputs[:, 1] + ystart

    return outputs[:, 0], outputs[:, 1], outputs[:, 2], outputs[:, 3], outputs[:, 4]

def contact_surface(x:np.array,y:np.array,
                    mean:np.float64,var:np.float64,
                    corl:np.array,
                    mask=None):
    '''
    Creates gaussian random contact surface with mean value and variance input
    with the spectral method from  Dietrich & Newsam (1993).
    Input:
    -------------
    x,y: 2D grid of x and y points
    mean: mean value
    var: variance
    corl: correlation lenghts (same unit as x and y) in x and y directions
    mask: mask array (same dimensions as x and y)
    Returns:
    Z: output np.array with same dimensions as x and y and with Z values corrensponding to the surface
    '''
    Z = specsim(x,y,mean=mean,var=var,corl=corl,mask=mask)
    return Z


