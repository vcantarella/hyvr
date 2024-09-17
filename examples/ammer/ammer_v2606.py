# %% [markdown]
# # Modelling the Tufa layer in the Ammer Valley Quaternary floodplain stratigraphy
#
# This notebook explain step-by-step how to create a object based sedimentary structure model using HyVR and the python environment. The model will represent the tufa layer in the Quaternary floodplain sediments of the Ammer Valler, Tuebingen, Germany.

# %% [markdown]
# ## Packages
#
# Besides HyVR, we use numpy for numerical computing. Since we are interested in using this model for posterior flow simulations, we will use the capabilities in flopy, specially for handling the grid and exporting VTKs (3D render models).

# %%
from hyvr.tools import ferguson_curve  # used for the channel curve creation
from hyvr import channel  # channel object creation
from hyvr import half_ellipsoid  # trough creation
import scipy  # general scientific calculations
import flopy  # our modelling interface
import numpy as np  # general numerical functions and array manipulation
from hyvr.utils import specsim
from hyvr.tools import specsim_surface
import numba

# %% [markdown]
# ## Grid/Model creation
#
# HyVR should work on any structured grid. One example would be creating a grid with `np.meshgrid`, the numpy function for grids. However, we are interested in flow simulations, and MODFLOW is the standard. The python interface, flopy has grid creation capabilities that can be easily translated to MODFLOW grids, thus we use that for our grid creation

# %%
# Model creation:
name = "ammer_V2606"
ws = "examples/ammer"
sim = flopy.mf6.MFSimulation(
    sim_name=name,
    exe_name="mf6",
    version="mf6",
    sim_ws=ws,
)
# Nam file
model_nam_file = "{}.nam".format(name)
# Groundwater flow object:
gwf = flopy.mf6.ModflowGwf(
    sim,
    modelname=name,
    model_nam_file=model_nam_file,
    save_flows=True,
)
# Grid properties:
Lx = 900  # problem lenght [m]
Ly = 600  # problem width [m]
H = 9  # aquifer height [m]
delx = 1.5  # block size x direction
dely = 1.5  # block size y direction
delz = 0.2  # block size z direction
nlay = int(H / delz)
ncol = int(Lx / delx)  # number of columns
nrow = int(Ly / dely)  # number of layers

# Flopy Discretizetion Objects (DIS)
dis = flopy.mf6.ModflowGwfdis(
    gwf,
    xorigin=0.0,
    yorigin=0.0,
    nlay=nlay,
    nrow=nrow,
    ncol=ncol,
    delr=dely,
    delc=delx,
    top=H,
    botm=np.arange(H - delz, 0 - delz, -delz),
)

# Node property flow
k = 1e-5  # Model conductivity in m/s
npf = flopy.mf6.ModflowGwfnpf(
    gwf,
    icelltype=0,  # This we define the model as confined
    k=k,
)

# Acessing the grid
grid = gwf.modelgrid

# cell centers
centers = grid.xyzcellcenters

X = centers[0]
Y = centers[1]
Z = centers[2]

# broadcasting the X, Y to the same shape as Z (full grid shape)
X = np.broadcast_to(X, Z.shape)
Y = np.broadcast_to(Y, Z.shape)

# %% [markdown]
# ## Modelling Sedimentary Structures
#
# We base the framework on the original work from Bennett et al (2018), which has written the first version of HyVR. The framework is hierarchical, meaning that structures can be organized in different scales depending on the context, but that they may represent complex sedimentary archictectures.
#
# From the analysis of the facies present in the tufa we came up with an achitecture formed by the following elements:
#
# The tufa is likely transported from upstream. Seepage from supersaturated groundwater from the carbonate formation intersects the ammer river upstream and probably caused the precipitation of tufa particles in association with adequate flora in a wetland-like environment. Then river water would carry this sediment downstream along with organic matter (often preserved as plant remains), and deposit this material in the Ammer floodplain. We see different facies with varying amounts of tufa and organic matter, and varying composition of organic matter. Likely the system stabilized for periods of time, giving structure to different local wetland environments. Depending on the stability and preservation characteristics peat lenses would form, and the river channel shape would be preserved as gravel deposits. During sediment inflow periods, continuous input of sedimentation would make the river shape unstable, meaning it would not preserve its shape. The wetland environments would then receive tufa sedimentation and deposit mixes of tufa clasts and phytoclasts and organic matter. We think of this period as a succession of discontinuous lenses which are elongated in the direction of flow of different facies associated with different local wetland environments reworked by transport and deposition of external tufa particles.
#
# Therefore, we can think of the system as an aggradational sequence of lenses of different mix composition between tufas and phytoclasts, organic particles, which comprises different facies recorded in the sedimentary analysis. At the end of the sequence we would have the reduction of sedimentary the sedimentary load, leading to a stable configuration where peat lenses would be preserved and the preservation of channel features. Upon the next increase of sedimentary load, the channel features would be preferentially filled with gravel particles, while the peat lenses then would be buried, and the sequence then repeats itself.
#
# The algorithm is organized as such:
#
# 1. Define the sequence thicknesses (sampled from a distribution)
# 2. Over the thickness t:
#     2.1. Iterate over each facies f associated with the aggradation period:
#         2.1.1. generate lens of thickness t at a randomly sampled location and with reasonably chosen dimensions. Assign it to facies f
#         2.1.2. repeat 2.1 until the proportion of facies f is slightly above the calculated proportion (since one object can erode the previous we use slightly bigger proportion in the algorithm).
#     2.3. Generate lens of thickness t or max_peat_thickness at a randomly sampled location and with reasonably chosen dimension of the facies peat
#     2.4. repeat 2.3 until proportion of peat is the same as the calculated proportion.
#     2.5. Generate a channel starting on the left of the grid and on a randomly sampled high y value with thickness t or max_channel_thickness and width $\approx$ 4 m.
#     2.6. Generate a channel startubg on the left of the grid (x=0) and on a randomly sampled low y values with thickness t or max_channel_thickness and width $\approx$ 4 m.
# 3. Add the base level by thickness t and repeat 2. unil the end of the sequence (7m high).
#
# %%
# Define the top and bottom of the model as the top and bottom of the sequence. Assign the unassigned cells to the facies 4 (background facies).

np.random.seed(37893)
mean_top = H - 1.86
var_top = 0.7
corl_top = np.array([70, 792])
surf_top = specsim(X[0, :, :], Y[0, :, :], mean=mean_top, var=var_top, corl=corl_top)

mean_botm = H - 8.3
var_botm = .9
corl_botm = np.array([300, 900])
surf_botm = specsim(
    X[0, :, :], Y[0, :, :], mean=mean_botm, var=var_botm, corl=corl_top
)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot( X[0, int(nrow/2), :],surf_top[int(nrow/2),:], label="Top")
ax.plot(X[0, int(nrow/2), :],surf_botm[int(nrow/2),:],  label="Botm")
ax.legend()
plt.show()
# %% [markdown]
surf_top2 = specsim_surface(X[0, :, :], Y[0, :, :], mean=mean_top, var=var_top, corl=corl_top)
surf_botm2 = specsim_surface(X[0, :, :], Y[0, :, :], mean=mean_botm, var=var_botm, corl=corl_top)
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot( X[0, int(nrow/2), :],surf_top[int(nrow/2),:], label="Top")
ax.plot( X[0, int(nrow/2), :],surf_top2[int(nrow/2),:], label="Top2")
ax.plot(X[0, int(nrow/2), :],surf_botm[int(nrow/2),:],  label="Botm")
ax.plot(X[0, int(nrow/2), :],surf_botm2[int(nrow/2),:],  label="Botm2")
ax.legend()
plt.show()


# %%
# ### Defining the sequence of thicknesses
#
# We will assume an average thickness of 0.7 m. The first layer in the system is modelled deterministically. It is $\approx$ 0.4 m thickness and composed with light color tufa with fossil and low organic matter content. The remaining layers are modelled probabilistically.
#  which means that in the sequence of 7 m, we randomly sample 10 thicknesses:
#
# Below we have calculated a distribution function that randomly sampled thicknesses with the characteristics above:

# %%
# according to answer in : https://math.stackexchange.com/questions/291174/probability-distribution-of-the-subinterval-lengths-from-a-random-interval-divis
# The cumulative distribution function for intervals randomly sampled from the interval [0,a] is:
simulated_thickness = (mean_top) - 1
n = 8
print(f"With the number of layers: {n}")
F = lambda t: 1 - ((simulated_thickness - t) / simulated_thickness) ** (n - 1)
# The probability density function is then:
f = (
    lambda t: (n - 1)
    / simulated_thickness
    * ((simulated_thickness - t) / simulated_thickness) ** (n - 2)
)
mean_thickness = scipy.integrate.quad(lambda t: t * f(t), 0, simulated_thickness)[0]
median_thickness = scipy.optimize.fsolve(lambda t: F(t) - 0.5, mean_thickness)[0]
# The median thickness would be:
print(f"The median thickness is:{median_thickness}")
# The mean thickness would be:
print(f"The mean thickness is:{mean_thickness}")

# %% [markdown]
# The code above has calculated that a 9 layer model would on average produce a thickness agreeable with measured average of 0.7 meters.
# To generate the model thicknesses we can just generate random samples on the interval from 0 to the total modelled thickness. Unfortunately, due to cell size restrictions, we cannot model layers that are too small (< 0.3 m), therefore we iterate unil we have a thickness model with layers that are bigger than 0.3:

# %%
np.random.seed(37893)
min_thick = 0
while min_thick < 0.3:
    zs = np.random.uniform(0, simulated_thickness, size=n - 1)
    ordered_zs = np.sort(zs)
    ordered_zs = np.append(ordered_zs, simulated_thickness)
    ordered_zs = np.append(0, ordered_zs)
    thicknesses = np.diff(ordered_zs)
    min_thick = np.min(thicknesses)

# %%
print(f"The thickness array is:{thicknesses})")
print(f"The minimum thickness is:{min_thick}")
print(f"The total thickness is:{np.sum(thicknesses)}")
print(f"The mean thickness is:{np.mean(thicknesses)}")
print(f"THe number of layers is:{len(thicknesses)}")

# %%
testf = np.empty((nrow, ncol), dtype=np.int32)
np.unique(testf)


# %% [markdown]
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
    distance = lambda X, x, y: np.sqrt((X[0] - x) ** 2 + (X[1] - y) ** 2)
    # compute distance
    d_array = np.zeros((P.shape[0]))
    glob_min_idx = np.zeros((P.shape[0]))
    for i in numba.prange(P.shape[0]):
        d_line = distance(P[i], x, y)
        d_array[i] = np.min(d_line)
        glob_min_idx[i] = np.argmin(d_line)
    return d_array, glob_min_idx


# %% [markdown]
np.random.seed(37893)
facies = np.zeros_like(Z, dtype=np.int32) + 7
dip = np.zeros_like(Z, dtype=np.float64)
dipdir = np.zeros_like(Z, dtype=np.float64)
thicknesses

## Running the lower sequence
x = X[0, :, :].ravel()
y = Y[0, :, :].ravel()
P = np.column_stack([x, y])
z_0 = 0
for thick in thicknesses:
    # creating anastamosing channel pattern:
    main_channels = []
    channels = []
    for i in range(6):
        ystart = np.random.uniform(0, 600)
        channel_curve = ferguson_curve(
            h=0.3,
            k=np.pi / 200,
            eps_factor=(np.pi / 1.5) ** 2,
            flow_angle=0.0,
            s_max=1500,
            xstart=-500,
            ystart=ystart,
        )
        main_channels.append(channel_curve)
        indexes = np.random.choice(
            np.arange(channel_curve[0].shape[0]), size=4, replace=False
        )
        xstart = channel_curve[0][indexes]
        ystart = channel_curve[1][indexes]
        for x, y in zip(xstart, ystart):
            channel_derived_channel = ferguson_curve(
                h=0.3,
                k=np.pi / 200,
                eps_factor=(np.pi / 1.5) ** 2,
                flow_angle=np.random.uniform(-np.pi / 18, np.pi / 18),
                s_max=1000,
                xstart=x,
                ystart=y,
            )
            channels.append(channel_derived_channel)
    total_channels = main_channels + channels
    min_distance_array = np.zeros((P.shape[0], len(total_channels)))
    for i, channel_ in enumerate(total_channels):
        min_distance_array[:, i], _ = min_distance(channel_[0], channel_[1], P)
    # cut the proportion to 30 % of the cells in the x, y plane
    min_arr = min_distance_array.min(axis=1)
    # Assuming `min_array` and `t` are already defined
    # Get the sorted indices of t
    sorted_indices = np.argsort(min_arr)
    # Calculate the number of indices to select (30% of the total)
    num_indices = int(len(sorted_indices) * 0.2)
    # Select the last 30% of the indices
    selected_indices = sorted_indices[-num_indices:]

    primitive = np.ones_like(min_arr, dtype=np.int32) * 7
    primitive = np.ravel(primitive)
    primitive[selected_indices] = 6
    primitive = primitive.reshape((Z.shape[1], Z.shape[2]))
    z_ = Z[:, 0, 0]
    for i in range(z_.shape[0]):
        if z_[i] >= z_0:
            facies[i, :, :] = primitive
    print(f"Finished the primitive of layer {z_0 + thick}")
    ## Adding ponds:
    p_ponds = 0
    logic_tufa = (Z >= z_0) & (Z <= z_0 + thick)
    while p_ponds < 0.30:
        x_c = np.random.uniform(0, 900)
        y_c = np.random.uniform(0, 600)
        z_c = z_0 + thick + np.random.uniform(0, 0.1)
        a = np.random.uniform(50, 80)
        b = np.random.uniform(30, 60)
        c = thick
        azim = np.random.uniform(20, -20)
        half_ellipsoid(
            facies,
            dip,
            dipdir,
            X,
            Y,
            Z,
            center_coords=np.array([x_c, y_c, z_c]),
            dims=np.array([a, b, c]),
            azim=azim,
            facies=np.array([2]),
        )
        # k[facies_trough != -1] = np.random.lognormal(mu_tufa, sigma=sigma_tufa)
        p_ponds = np.sum(facies[logic_tufa] == 2) / np.sum(logic_tufa)
    print(f"Finished ponds in layer {z_0 + thick}")
    ## Adding the channels, the ponds and the peat lenses
    for channel_ in main_channels:
        channel(
            facies,
            dip,
            dipdir,
            X,
            Y,
            Z,
            z_top=z_0 + thick,
            curve=np.c_[channel_[0], channel_[1]],
            parabola_pars=np.array([30, thick]),
            facies=np.array([4]),
        )
    for channel_ in channels:
        channel(
            facies,
            dip,
            dipdir,
            X,
            Y,
            Z,
            z_top=z_0 + thick,
            curve=np.c_[channel_[0], channel_[1]],
            parabola_pars=np.array([20, thick]),
            facies=np.array([4]),
        )
    print(f"Finished channels in layer {z_0 + thick}")
    ## assiging peat to areas close to water bodies:
    water_bodies = (facies == 2) | (facies == 4)
    z_ = Z[:, 0, 0]
    ind = np.where(z_ <= z_0 + thick)
    layer = np.min(ind[0])
    layer
    water_bodies = water_bodies[layer, :, :]
    xs = X[layer, :, :][water_bodies]
    ys = Y[layer, :, :][water_bodies]
    xs.shape
    ys.shape
    index_ = np.random.choice(np.arange(0, xs.shape[0]))
    ## Adding peat lenses:
    p_peat = 0
    if thick > 0.4:
        c = 0.4
    else:
        c = thick
    while p_peat < 0.20:
        index_ = np.random.choice(np.arange(0, xs.shape[0]))
        x_c = xs[index_]
        y_c = ys[index_]
        z_c = z_0 + thick
        a = np.random.uniform(30, 60)
        b = np.random.uniform(20, 40)
        c = c
        azim = np.random.uniform(-20, 20)
        facies_code = np.random.choice([8, 9])
        half_ellipsoid(
            facies,
            dip,
            dipdir,
            X,
            Y,
            Z,
            center_coords=np.array([x_c, y_c, z_c]),
            dims=np.array([a, b, c]),
            azim=azim,
            facies=np.array([facies_code]),
        )
        # k[facies_trough != -1] = np.random.lognormal(mu_tufa, sigma=sigma_tufa)
        logic_tufa = (Z >= z_0 + thick - c) & (Z <= z_0 + thick)
        p_peat = np.sum(facies[logic_tufa] == facies_code) / np.sum(logic_tufa)
    print(f"Finished the layer {z_0 + thick}")
    # resetting z_0:
    z_0 += thick

# %% [markdown]
# Creating the final layer (the top deterministic layer) and assigning unassigned cells to the facies 4
np.random.seed(37893)
min_height = z_0
facies[Z > min_height] = 10
heights = np.arange(min_height, np.max(surf_top), 0.05)
x_c = -np.random.uniform(200, 300)
y_c = np.random.uniform(200, 600)
x = X[0, :, :].ravel()
y = Y[0, :, :].ravel()
P_0 = np.column_stack([x, y])

heights
# %%
# final_layer!
for height in heights:
    # adding initial channel
    size_indexes = 0
    while size_indexes < 200:
        gravel_channel = ferguson_curve(
            h=0.3,
            k=np.pi / 200,
            eps_factor=(np.pi) ** 2,
            flow_angle=0.0,
            s_max=1500 - x_c,
            xstart=x_c,
            ystart=y_c,
        )
        min_distance_arr, _ = min_distance(gravel_channel[0], gravel_channel[1], P_0)
        indexes = np.where(min_distance_arr < 200)
        indexes = indexes[0]
        xs = x[indexes]
        ys = y[indexes]
        size_indexes = xs.shape[0]
    print("adding peat lenses")
    p_tufa = 0
    thick = 0.2
    logic_tufa = (Z >= height) & (Z <= height + thick)
    # calculating distance to the channel:
    xp = X[logic_tufa].ravel()
    yp = Y[logic_tufa].ravel()
    P = np.column_stack([xp, yp])
    min_distance_arr, _ = min_distance(gravel_channel[0], gravel_channel[1], P)
    nindexes = np.where(min_distance_arr < 200)
    nindexes = nindexes[0]
    while p_tufa < 0.90:
        index_ = np.random.choice(np.arange(0, xs.shape[0]))
        x_t = xs[index_]
        y_t = ys[index_]
        z_t = height + thick
        a = np.random.uniform(60, 90)
        b = np.random.uniform(40, 50)
        c = np.random.uniform(thick, thick + 0.2)
        azim = np.random.uniform(-20, 20)
        half_ellipsoid(
            facies,
            dip,
            dipdir,
            X,
            Y,
            Z,
            center_coords=np.array([x_t, y_t, z_t]),
            dims=np.array([a, b, c]),
            azim=azim,
            facies=np.array([11]),
        )
        p_tufa = np.sum(facies[logic_tufa].ravel()[nindexes] == 11) / np.sum(
            logic_tufa[logic_tufa].ravel()[nindexes]
        )
        print(p_tufa)
    ## Add channel facies:
    channel(
        facies,
        dip,
        dipdir,
        X,
        Y,
        Z,
        z_top=height + thick,
        curve=np.c_[gravel_channel[0], gravel_channel[1]],
        parabola_pars=np.array([25, thick + 0.2]),
        facies=np.array([12]),
    )
    print(f"Finished the layer {height}")


# %% [markdown]
# masking the model at the top and bottom:
facies[Z >= surf_top] = 21
facies[Z <= surf_botm] = 31
# %%
# export numpy model:
centroid = np.stack([X, Y, Z])
np.save(
    "C:\\Users\\vcant\OneDrive\\Kassel\\ammer_field\\hyvr_models\\facies_v7.npy", facies
)
np.save(
    "C:\\Users\\vcant\OneDrive\\Kassel\\ammer_field\\hyvr_models\\centroid_v7.npy",
    centroid,
)

# %%
