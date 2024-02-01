import flopy
import numpy as np
import scipy
from hyvr.objects.channel import channel
from hyvr.objects.trough import trough
from hyvr.tools import ferguson_curve

# Model creation:
name = "ammer_simplified"
ws = "."
sim = flopy.mf6.MFSimulation(
    sim_name=name,
    exe_name="mf6",
    version="mf6",
    sim_ws=ws,
)
# Simulation time:
tdis = flopy.mf6.ModflowTdis(
    sim, pname="tdis", time_units="DAYS", nper=1, perioddata=[(1.0, 1, 1.0)]
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
Lx = 200  # problem lenght [m]
Ly = 100  # problem width [m]
H = 5  # aquifer height [m]
delx = 1  # block size x direction
dely = 1  # block size y direction
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
    top=7.0,
    botm=np.arange(H - delz, 0 - delz, -delz),
)

# Flopy initial Conditions
h0 = 18
start = h0 * np.ones((nlay, nrow, ncol))
ic = flopy.mf6.ModflowGwfic(gwf, pname="ic", strt=start)

# Node property flow
k = 1e-5  # Model conductivity in m/s
npf = flopy.mf6.ModflowGwfnpf(
    gwf,
    icelltype=1,  # This we define the model as convertible (water table aquifer)
    k=k,
)
# boundary conditions:
grid = gwf.modelgrid

centers = grid.xyzcellcenters

X = centers[0]
Y = centers[1]
Z = centers[2]

X = np.broadcast_to(X, Z.shape)
Y = np.broadcast_to(Y, Z.shape)


# print(channel_curve_1)
# fig,ax = plt.subplots()
# ax.plot(channel_curve_1[0],channel_curve_1[1])
# fig.show()

# building the layers:
# according to answer in : https://math.stackexchange.com/questions/291174/probability-distribution-of-the-subinterval-lengths-from-a-random-interval-divis
# The cumulative distribution function for intervals randomly sampled from the interval [0,a] is:
simulated_thickness = 5
n = np.floor(simulated_thickness / 0.6)
F = lambda t: 1 - ((simulated_thickness - t) / simulated_thickness) ** (n - 1)
F_ = lambda t, p: F(t) - p
thicks = []
sum_thick = 0.0
while sum_thick < simulated_thickness:
    p = np.random.uniform(0, 1)
    sol = scipy.optimize.root(F_, 0.1, args=(p,))
    thick = sol.x[0]
    if thick < 0.3:
        thick = 0.3
    sum_thick += thick
    thicks.append(thick)

print(sum_thick)

thicks = [thick for thick in thicks if (thick < simulated_thickness)]

# We are simulating the facies:
##TUFA:
# T8+T9: PEAT
# T11: gravel
colors = {
    1: "#FCF2A8",
    2: "#cec47f",
    4: "#c9c186",
    5: "#7e7954",
    6: "#b0b468",
    7: "#c8b23d",
    8: "#323021",
    11: "#b7b7b7",
}

facies_tufa = np.array([2, 3, 4, 5], dtype=np.int32)
facies = np.empty_like(Z, dtype=np.int32)
z_0 = 0.0
layer = 0
for thick in thicks:
    # peat lenses:
    peat = 0.0
    # peat lenses
    while peat < 0.15:
        x_c = np.random.uniform(0, 200)
        y_c = np.random.uniform(0, 100)
        z_c = z_0 + thick + np.random.uniform(-0.2, 0)
        a = np.random.uniform(20, 50)
        b = np.random.uniform(15, 25)
        azim = np.random.uniform(60, 120)
        if thick > 0.6:
            peat_depth = 0.6
        else:
            peat_depth = thick
        c = peat_depth

        facies_trough, dip_dir_trough, dip_trough = trough(
            X,
            Y,
            Z,
            center_coords=np.array([x_c, y_c, z_c]),
            dims=np.array([a, b, c]),
            azim=azim,
            facies=np.array([8]),
        )
        facies[facies_trough != -1] = facies_trough[facies_trough != -1]
        logic_peat = (Z >= z_0) & (Z <= z_0 + thick)
        peat = np.sum(facies[logic_peat] == 8) / np.sum(logic_peat)
        print(peat)
    # channels
    channel_curve_1 = ferguson_curve(
        h=0.1,
        k=np.pi / 200,
        eps_factor=(np.pi / 1.5) ** 2,
        flow_angle=0.0,
        s_max=400,
        xstart=0,
        ystart=25,
    )
    y_shift_1 = np.random.uniform(-25, 25)
    channel_1 = np.c_[channel_curve_1[0], channel_curve_1[1] + y_shift_1]
    depth = np.random.uniform(0.5, 0.6)
    channel_f, channel_dip_dir, channel_dip = channel(
        X,
        Y,
        Z,
        z_top=z_0 + thick,
        curve=channel_1,
        parabola_pars=np.array([4, depth]),
        facies=np.array([11]),
    )
    facies[channel_f != -1] = channel_f[channel_f != -1]
    # resetting z_0:
    z_0 += thick

    vtk = flopy.export.vtk.Vtk(model=gwf, modelgrid=grid)
    vtk.add_array(facies, "facies")
    vtk.write(f"ammer_lay{layer}")
    layer += 1


# final_layer!
facies[Z > simulated_thickness] = 1  # T1 facies
