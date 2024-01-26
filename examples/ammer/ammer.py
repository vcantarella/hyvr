import hyvr
from hyvr.utils import ferguson_curve
from hyvr.objects.channel import channel
from hyvr.objects.trough import trough
from hyvr.objects.sheet import sheet
import scipy
import flopy
import numpy as np
import os
import matplotlib.pyplot as plt

# Model creation:
name = 'ammer_V0'
directory = 'examples'
ws = os.path.join(directory,"ammer")
sim = flopy.mf6.MFSimulation(
    sim_name=name, exe_name="mf6", version="mf6", sim_ws=ws,
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
Lx = 2000 #problem lenght [m]
Ly = 600 #problem width [m]
H = 7  #aquifer height [m]
delx = 1.5 #block size x direction
dely = 1.5 #block size y direction
delz = 0.2 #block size z direction
nlay = int(H/delz)
ncol = int(Lx/delx) # number of columns
nrow = int(Ly/dely) # number of layers

# Flopy Discretizetion Objects (DIS)
dis = flopy.mf6.ModflowGwfdis(
    gwf,
    xorigin=0.,
    yorigin=0.,
    nlay=nlay,
    nrow=nrow,
    ncol=ncol,
    delr=dely,
    delc=delx,
    top=7.,
    botm=np.arange(H-delz,0-delz,-delz),
)

# Flopy initial Conditions
h0 = 18
start = h0 * np.ones((nlay, nrow, ncol))
ic = flopy.mf6.ModflowGwfic(gwf, pname="ic", strt=start)

# Node property flow
k = 1e-5 # Model conductivity in m/s
npf = flopy.mf6.ModflowGwfnpf(
    gwf,
    icelltype=1, #This we define the model as convertible (water table aquifer)
    k=k,
)
#boundary conditions:
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
a = 7-0.4
n = np.floor(a/0.7)
F = lambda t: 1-((a-t)/a)**(n-1)
F_ = lambda t, p: F(t)-p
thicks = []
sum_thick = 0.
while sum_thick < a:
    p = np.random.uniform(0,1)
    sol = scipy.optimize.root(F_, 0.1, args = (p,))
    thick = sol.x[0]
    if thick < 0.3:
        thick = 0.3
    sum_thick += thick
    thicks.append(thick)


#We are simulating the facies:
##TUFA: 
    #T1: upper layer (first 40 cm): white tufa with fossils
    #T2+T3: tufa with organics + plant remains
    #T4: white no fossil
    #T5: brown tufa + organic matter and plant remains (darker and wetter than T2,T3)
    #T6: moss
    #T7: sticks + organics
    #T8+T9: PEAT
    #T11: gravel
colors = {1:"#FCF2A8",2:"#cec47f",4:"#c9c186",5:"#7e7954",6:"#b0b468",7:"#c8b23d",8:"#323021",11:"#b7b7b7"}

facies_tufa = np.array([2,3,4,5], dtype=np.int32)
facies = np.empty_like(Z,dtype=np.int32)
z_0 = 0.
for thick in thicks:
    ## initial tufa sheets: they are modelled as very elongated ellipsoids representing discontinuous layers
    p_t23 = 0
    while p_t23 < 0.3:
        x_c = np.random.uniform(0,2000)
        y_c = np.random.uniform(0,600)
        z_c = z_0+thick + np.random.uniform(-0.2,0)
        a = np.random.uniform(300,600)
        b = np.random.uniform(150,300)
        c = thick
        azim = np.random.uniform(70,110)
        facies_trough, dip_dir_trough, dip_trough = trough(X,Y,Z,center_coords=np.array([x_c,y_c,z_c]),
                                                        dims = np.array([a,b,c]),azim = azim,facies=np.array([2]))
        facies[facies_trough!=-1] = facies_trough[facies_trough!=-1]
        logic_tufa = (Z >= z_0) & (Z <= z_0+thick)
        p_t23 = np.sum(facies[logic_tufa] == 2)/np.sum(logic_tufa)
    p_t4 = 0
    while p_t4 < 0.4:
        x_c = np.random.uniform(0,2000)
        y_c = np.random.uniform(0,600)
        z_c = z_0+thick + np.random.uniform(-0.2,0)
        a = np.random.uniform(300,600)
        b = np.random.uniform(150,300)
        c = thick
        azim = np.random.uniform(70,110)
        facies_trough, dip_dir_trough, dip_trough = trough(X,Y,Z,center_coords=np.array([x_c,y_c,z_c]),
                                                        dims = np.array([a,b,c]),azim = azim,facies=np.array([4]))
        facies[facies_trough!=-1] = facies_trough[facies_trough!=-1]
        logic_tufa = (Z >= z_0) & (Z <= z_0+thick)
        p_t4 = np.sum(facies[logic_tufa] == 4)/np.sum(logic_tufa)
    p_t5 = 0
    while p_t5 < 0.3:
        x_c = np.random.uniform(0,2000)
        y_c = np.random.uniform(0,600)
        z_c = z_0+thick + np.random.uniform(-0.2,0)
        a = np.random.uniform(300,600)
        b = np.random.uniform(150,300)
        c = thick #thickness until the original base (more or less)
        azim = np.random.uniform(70,110)
        facies_trough, dip_dir_trough, dip_trough = trough(X,Y,Z,center_coords=np.array([x_c,y_c,z_c]),
                                                        dims = np.array([a,b,c]),azim = azim,facies=np.array([5]))
        facies[facies_trough!=-1] = facies_trough[facies_trough!=-1]
        logic_tufa = (Z >= z_0) & (Z <= z_0+thick)
        p_t5 = np.sum(facies[logic_tufa] == 5)/np.sum(logic_tufa)
    p_t6 = 0
    while p_t6 < 0.2:
        x_c = np.random.uniform(0,2000)
        y_c = np.random.uniform(0,600)
        z_c = z_0+thick + np.random.uniform(-0.2,0)
        a = np.random.uniform(300,600)
        b = np.random.uniform(150,300)
        c = thick #thickness until the original base (more or less)
        azim = np.random.uniform(70,110)
        facies_trough, dip_dir_trough, dip_trough = trough(X,Y,Z,center_coords=np.array([x_c,y_c,z_c]),
                                                        dims = np.array([a,b,c]),azim = azim,facies=np.array([6]))
        facies[facies_trough!=-1] = facies_trough[facies_trough!=-1]
        logic_tufa = (Z >= z_0) & (Z <= z_0+thick)
        p_t6 = np.sum(facies[logic_tufa] == 6)/np.sum(logic_tufa)
    p_t7 = 0
    while p_t7 < 0.3:
        x_c = np.random.uniform(0,2000)
        y_c = np.random.uniform(0,600)
        z_c = z_0+thick + np.random.uniform(-0.2,0)
        a = np.random.uniform(300,600)
        b = np.random.uniform(150,300)
        c = thick #thickness until the original base (more or less)
        azim = np.random.uniform(70,110)
        facies_trough, dip_dir_trough, dip_trough = trough(X,Y,Z,center_coords=np.array([x_c,y_c,z_c]),
                                                        dims = np.array([a,b,c]),azim = azim,facies=np.array([7]))
        facies[facies_trough!=-1] = facies_trough[facies_trough!=-1]
        logic_tufa = (Z >= z_0) & (Z <= z_0+thick)
        p_t7 = np.sum(facies[logic_tufa] == 7)/np.sum(logic_tufa)
    

    #peat lenses:
    peat = 0.
    #peat lenses
    while peat < 0.15:
        x_c = np.random.uniform(0,2000)
        y_c = np.random.uniform(0,600)
        z_c = z_0+thick+ np.random.uniform(-0.2,0)
        a = np.random.uniform(200,300)
        b = np.random.uniform(100,150)
        azim = np.random.uniform(60,120)
        if thick > 0.7:
            peat_depth = 0.7
        else:
            peat_depth = thick
        c = peat_depth

        facies_trough, dip_dir_trough, dip_trough = trough(X,Y,Z,center_coords=np.array([x_c,y_c,z_c]),
                                                        dims = np.array([a,b,c]),azim = azim,facies=np.array([8]))
        facies[facies_trough!=-1] = facies_trough[facies_trough!=-1]
        logic_peat = (Z >= z_0) & (Z <= z_0+thick)
        peat = np.sum(facies[logic_peat] == 8)/np.sum(logic_peat)
        print(peat)
    #channels
    channel_curve_1 = ferguson_curve(h=.1,k=np.pi/200,eps_factor=(np.pi/1.5)**2,flow_angle=0.,
                                s_max= 4000, xstart=0,ystart=25)
    y_shift_1 = np.random.uniform(400,500)
    channel_1 = np.c_[channel_curve_1[0],channel_curve_1[1]+y_shift_1]
    if thick > 0.6:
        depth = 0.6
    else:
        depth = thick
    channel_f, channel_dip_dir, channel_dip = channel(X,Y,Z,z_top=z_0+thick, curve = channel_1,
                                                    parabola_pars=np.array([4,depth]),facies=np.array([11]))
    facies[channel_f!=-1] = channel_f[channel_f!=-1]
    channel_curve_2 = ferguson_curve(h=.1,k=np.pi/200,eps_factor=(np.pi/1.5)**2,flow_angle=0.,
                                s_max= 4000, xstart=0,ystart=25)
    y_shift_2 = np.random.uniform(40,150)
    channel_2 = np.c_[channel_curve_2[0],channel_curve_2[1]+y_shift_2]
    channel_f, channel_dip_dir, channel_dip = channel(X,Y,Z,z_top=z_0+thick, curve = channel_2,
                                                    parabola_pars=np.array([4,depth]),facies=np.array([11]))
    facies[channel_f!=-1] = channel_f[channel_f!=-1]

    #resetting z_0:
    z_0 += thick

#final_layer!
facies[Z >= a] = 1 #T1 facies
#Why this doenst work in the script??

print(np.sum(facies == -1)/facies.size)
print(np.sum(facies==0)/facies.size)
facies[facies == -1] = 2
facies[facies == 0] = 2



vtk = flopy.export.vtk.Vtk(model=gwf, modelgrid=grid)

vtk.add_array(facies, 'facies')

vtk.write("ammer_v2")

np.save("facies_ammer_v2.npy",facies)
np.save("facies_Z_v2.npy",Z)

