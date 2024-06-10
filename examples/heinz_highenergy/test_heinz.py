import numpy as np
import numba
from src.hyvr.objects.trough import half_ellipsoid
from src.hyvr.objects.sheet import sheet
from src.hyvr.utils import azimuth_to_counter_clockwise, coterminal_angle

rng = np.random.default_rng(seed=42)

# Grid properties:
Lx = 900 #problem lenght [m]
Ly = 600 #problem width [m]
H = 7  #aquifer height [m]
delx = 0.5 #block size x direction
dely = 0.5 #block size y direction
delz = 0.1 #block size z direction
nlay = int(H/delz)
ncol = int(Lx/delx) # number of columns
nrow = int(Ly/dely) # number of layers

#creating a grid according to the MODFLOW convention
xs = np.linspace(0,Lx,ncol)
ys = np.linspace(0,Ly,nrow)
zs = np.linspace(0,H,nlay)
z,y,x = np.meshgrid(zs,ys,xs,indexing='ij')
z = np.flip(z,axis=0)
y = np.flip(y,axis=1)

# Defining the facies:
facies_list = np.arange(0, 4)
# Defining the facies names:
facies_names = ['S-x', 'GS-x', 'Gcm', 'Gcg,a']

# defining the arrays for the facies, dip direction and dip:
facies = np.ones(x.shape, dtype=np.int32) * (-1)
dip_dir = np.empty(x.shape)
dip = np.empty(x.shape)


# High discharge zone: large migrating scout pools with occasional
#  (but heavily eroded) channel beds

## Base layer:
xmin = 0
xmax = Lx
ymin = 0
ymax = Ly
bottom_surface = 0*np.ones_like(x[0,:,:])
top_surface = 0.5*np.ones_like(x[0,:,:])
dip_direction = coterminal_angle(azimuth_to_counter_clockwise(np.random.uniform(-10, 10)))
dip_angle = np.random.uniform(3, 30)
sheet(facies, dip, dip_dir,
        x,y,z,xmin,xmax,ymin,ymax,
                                 bottom_surface,
                                 top_surface,
                                 facies=np.array([1,2,3]),
                                 internal_layering=True,
                                 alternating_facies=True,
                                 dip = dip_angle,
                                 dip_dir=dip_direction,
                                 layer_dist=0.2)

# Randomnly adding sheets and troughs:
random_choice = np.array([0, 1]) # 0: sandy sheet, 1: massive gravel sheet
z_top = 0.5
# random_first trough depth
random_depth = np.random.uniform(1, 2)
previous_random_depth = 0.
# estimating the number of troughs: vol density = 90%
trough_volume = 20 * 9 * 1.5
total_volume = Lx * Ly * H
migration_number = 3
n_troughs = np.floor(total_volume / trough_volume / migration_number).astype(np.int32)

#jitted function to create troughs in parallel
@numba.jit(nopython=True, parallel=True)
def create_troughs(facies, dip, dip_dir, n_troughs, top):
    for j in numba.prange(n_troughs):
            # create the first trough:
            x_j = np.random.uniform(2, Lx-2)
            y_j = np.random.uniform(2, Lx-2)
            a = np.random.uniform(15, 24)
            b = np.random.uniform(6, 12)
            c = random_depth
            center_coords = np.array([x_j, y_j, top])
            model_azimuth = coterminal_angle(azimuth_to_counter_clockwise(np.random.uniform(-10, 10)))
            half_ellipsoid(facies, dip, dip_dir,x,y,z,center_coords=center_coords, dims=np.array([a, b, c]),
             facies=facies_list, internal_layering=True, alternating_facies=True,
             azim=model_azimuth,
             dip= np.random.uniform(3, 12),
             dip_dir=coterminal_angle(azimuth_to_counter_clockwise(np.random.uniform(-10, 10))),
             layer_dist=0.2)
            # migrating troughs:
            migration_distance = 5  # [m]
            x_i = x_j
            y_i = y_j
            for i in range(migration_number - 1):
                rad = model_azimuth * np.pi / 180
                x_i = x_i + migration_distance * np.cos(rad)
                y_i = y_i + migration_distance * np.sin(rad)
                if (x_i > Lx) | (y_i > Ly):
                    break
                if (x_i < 0) | (y_i < 0):
                    break
                center_coords = np.array([x_i, y_i, z_top])
                half_ellipsoid(facies, dip, dip_dir,x,y,z, center_coords=center_coords, dims=np.array([a, b, c]),
                    facies=facies_list, internal_layering=True, alternating_facies=True,
                    azim=model_azimuth,
                    dip=np.random.uniform(3, 12),
                    dip_dir=coterminal_angle(azimuth_to_counter_clockwise(np.random.uniform(-10, 10))),
                    layer_dist=0.2)




print(f'The number of Troughs is: {n_troughs}')
while z_top < H:
    print(z_top)
    choice = np.random.choice(random_choice)
    if choice == 0: #sheet
        thick = np.random.uniform(1, 0.5)
        bottom_surface = z_top*np.ones_like(x[0,:,:])
        top_surface = (z_top+thick)*np.ones_like(x[0,:,:])
        dip_direction = coterminal_angle(azimuth_to_counter_clockwise(np.random.uniform(-10, 10)))
        dip_angle = np.random.uniform(3, 10)
        sheet(facies, dip, dip_dir,
                x,y,z,xmin,xmax,ymin,ymax,
                                 bottom_surface,
                                 top_surface,
                                 facies=np.array([0,1,3]),
                                 internal_layering=True,
                                 alternating_facies=True,
                                 dip = dip_angle,
                                 dip_dir=dip_direction,
                                 layer_dist=0.2)
    else: #massive gravel sheet
        thick = np.random.uniform(0.8, 1.5)
        bottom_surface = z_top*np.ones_like(x[0,:,:])
        top_surface = (z_top+thick)*np.ones_like(x[0,:,:])
        sheet(facies, dip, dip_dir,
                x,y,z,xmin,xmax,ymin,ymax,
                                 bottom_surface,
                                 top_surface,
                                 facies=np.array([2]),
                                 internal_layering=False)
    z_top += thick
    if z_top > random_depth + previous_random_depth:
        previous_random_depth += random_depth
        create_troughs(facies, dip, dip_dir, n_troughs, previous_random_depth)
        random_depth = np.random.uniform(1, 2)



np.save('heinz_facies.npy', facies)
np.save('heinz_dip.npy', dip)
np.save('heinz_dip_dir.npy', dip_dir)

