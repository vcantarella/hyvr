import numpy as np
from src.hyvr.objects.trough import half_ellipsoid
from src.hyvr.objects.sheet import sheet
from src.hyvr.objects.channel import channel

def test_trough_volume():
    """
    Tests the area of a trough in a dense grid.
    """
    #creating a grid according to the MODFLOW convention
    xs = np.linspace(0,100,200)
    ys = np.linspace(0,80,160)
    zs = np.linspace(0,20,50)
    z,y,x = np.meshgrid(zs,ys,xs,indexing='ij')
    z = np.flip(z,axis=0)
    y = np.flip(y,axis=1)
    centercoord = np.array([50,40,19])
    dims = np.array([40,20,10])
    facies = np.ones_like(x, dtype=np.int32) * (-1)
    dip_array = np.empty_like(x)
    dip_dir_array = np.empty_like(x)
    half_ellipsoid(facies,dip_array,dip_dir_array,
        x,y,z,centercoord,
                                                      dims,
                                                      azim = 20.,facies = np.array([1]),
                                                      )
    area_meas = np.sum(facies == 1)*np.diff(xs)[0]*np.diff(ys)[0]*np.diff(zs)[0]
    area_true = np.pi*4/3*dims[0]*dims[1]*dims[2]/2
    assert np.abs(area_meas-area_true) < area_true*0.01

def test_bulb_volume():
    """
    Tests the area of a trough in a dense grid.
    """
    #creating a grid according to the MODFLOW convention
    xs = np.linspace(0,100,200)
    ys = np.linspace(0,80,160)
    zs = np.linspace(0,20,50)
    z,y,x = np.meshgrid(zs,ys,xs,indexing='ij')
    z = np.flip(z,axis=0)
    y = np.flip(y,axis=1)
    centercoord = np.array([50,40,19])
    dims = np.array([40,20,9])
    facies = np.ones_like(x, dtype=np.int32) * (-1)
    dip_array = np.empty_like(x)
    dip_dir_array = np.empty_like(x)
    half_ellipsoid(facies, dip_array, dip_dir_array,
    x,y,z,centercoord,
                                                      dims,
                                                      azim = 20.,facies = np.array([1,2,3]),
                                                      internal_layering=True,
                                                      alternating_facies=True,
                                                      bulb=True,
                                                      layer_dist=3,
                                                      dip_dir=20.,
                                                      )
   
    area_1 = np.sum(facies == 1)*np.diff(xs)[0]*np.diff(ys)[0]*np.diff(zs)[0]
    area_2 = np.sum(facies==2)*np.diff(xs)[0]*np.diff(ys)[0]*np.diff(zs)[0]
    area_3 = np.sum(facies==3)*np.diff(xs)[0]*np.diff(ys)[0]*np.diff(zs)[0]
    
    area_t = np.pi*4/3*dims[0]*dims[1]*dims[2]/2
    assert np.abs(area_1+area_2+area_3-area_t) < area_t*0.01

def test_trough_layer_volume():
    """
    Tests the area of a trough in a dense grid.
    """
    #creating a grid according to the MODFLOW convention
    xs = np.linspace(0,100,200)
    ys = np.linspace(0,80,160)
    zs = np.linspace(0,20,50)
    z,y,x = np.meshgrid(zs,ys,xs,indexing='ij')
    z = np.flip(z,axis=0)
    y = np.flip(y,axis=1)
    centercoord = np.array([50,40,19])
    dims = np.array([40,20,9])
    facies = np.ones_like(x, dtype=np.int32) * (-1)
    dip_array = np.zeros_like(x)
    dip_dir_array = np.zeros_like(x)
    half_ellipsoid(facies, dip_array, dip_dir_array,
        x,y,z,centercoord,
                                                      dims,
                                                      azim = 20.,facies = np.array([1,2,3]),
                                                      internal_layering=True,
                                                      alternating_facies=True,
                                                      bulb=False,
                                                      layer_dist=3,
                                                      dip_dir=20.,
                                                      )
   
    area_1 = np.sum(facies == 1)*np.diff(xs)[0]*np.diff(ys)[0]*np.diff(zs)[0]
    area_2 = np.sum(facies==2)*np.diff(xs)[0]*np.diff(ys)[0]*np.diff(zs)[0]
    area_3 = np.sum(facies==3)*np.diff(xs)[0]*np.diff(ys)[0]*np.diff(zs)[0]
    
    area_t = np.pi*4/3*dims[0]*dims[1]*9/2
    assert np.abs(area_1+area_2+area_3-area_t) < area_t*0.01
    assert np.allclose(np.nanmax(dip_dir_array), np.deg2rad(20))
   
def test_sheet_volume():
    xs = np.linspace(0,100,200)
    ys = np.linspace(0,80,160)
    zs = np.linspace(0,20,50)
    xmin = np.min(xs)
    xmax = np.max(xs)
    ymin = np.min(ys)
    ymax = np.max(ys)
    z,y,x = np.meshgrid(zs,ys,xs,indexing='ij')
    z = np.flip(z,axis=0)
    y = np.flip(y,axis=1)
    bottom_surface = y[0,:,:]*5/ymax+5
    top_surface = y[0,:,:]*-5/ymax+15
    facies = np.ones_like(x, dtype=np.int32) * (-1)
    dip_array = np.empty_like(x)
    dip_dir_array = np.empty_like(x)
    sheet(facies, dip_array, dip_dir_array,
    x,y,z,xmin,xmax,ymin,ymax,
                                 bottom_surface,
                                 top_surface,
                                 facies=np.array([1]))
    area_meas = np.sum(facies == 1)*np.diff(xs)[0]*np.diff(ys)[0]*np.diff(zs)[0]
    area_true = np.sum((top_surface-bottom_surface)*np.diff(xs)[0]*np.diff(ys)[0])
    assert np.abs(area_meas-area_true) < area_true*0.01

def test_sheet_layer_volume():
    xs = np.linspace(0,100,200)
    ys = np.linspace(0,80,160)
    zs = np.linspace(0,20,50)
    xmin = np.min(xs)
    xmax = np.max(xs)
    ymin = np.min(ys)
    ymax = np.max(ys)
    z,y,x = np.meshgrid(zs,ys,xs,indexing='ij')
    z = np.flip(z,axis=0)
    y = np.flip(y,axis=1)
    bottom_surface = y[0,:,:]*5/ymax+5
    top_surface = y[0,:,:]*-5/ymax+15
    facies = np.ones_like(x, dtype=np.int32) * (-1)
    dip_array = np.zeros_like(x)
    dip_dir_array = np.zeros_like(x)
    sheet(facies, dip_array, dip_dir_array,
        x,y,z,xmin,xmax,ymin,ymax,
                                 bottom_surface,
                                 top_surface,
                                 facies=np.array([1,2,3]),
                                 internal_layering=True,
                                 dip = 30,
                                 dip_dir=45,
                                 layer_dist=4)
    ars =np.sum(facies == 1) + np.sum(facies == 2) + np.sum(facies == 3)
    area_meas = ars*np.diff(xs)[0]*np.diff(ys)[0]*np.diff(zs)[0]
    area_true = np.sum((top_surface-bottom_surface)*np.diff(xs)[0]*np.diff(ys)[0])
    assert np.abs(area_meas-area_true) < area_true*0.01
    assert np.allclose(np.nanmax(dip_array), np.deg2rad(30))
    assert np.allclose(np.nanmax(dip_dir_array), np.deg2rad(45))



def test_straight_channel():
    xs = np.linspace(0,100,200)
    ys = np.linspace(0,80,160)
    zs = np.linspace(0,20,50)
    z,y,x = np.meshgrid(zs,ys,xs,indexing='ij')
    z = np.flip(z,axis=0)
    y = np.flip(y,axis=1)
    dims = np.array([20,7])
    curve_x = np.arange(0,100,0.01)
    curve_y = np.repeat(30,curve_x.size)
    curve = np.c_[curve_x,curve_y]
    facies = np.ones_like(x, dtype=np.int32) * (-1)
    dip_array = np.empty_like(x)
    dip_dir_array = np.empty_like(x)
    channel(facies, dip_array, dip_dir_array,
    x,y,z,z_top=10,
                                   curve=curve,
                                   parabola_pars=dims,
                                   facies=np.array([1]))
    ars =np.sum(facies == 1)
    area_meas = ars*np.diff(xs)[0]*np.diff(ys)[0]*np.diff(zs)[0]
    area_true = 2/3*dims[0]*dims[1]*100
    assert np.abs(area_meas-area_true) < area_true*0.01 