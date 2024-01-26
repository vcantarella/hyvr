.. _inout:

==========================================================
HyVR inputs and outputs
=========================================================

----------------------------------------------------------------------
Basic Usage
----------------------------------------------------------------------

The original project had an input file and many forms of outputs that could be specified.
Unfortunately, this becomes cumbersome to mantain, and often unecessary.
In this fork, the philosophy is that there are much better tools in the python ecosystem
to do different tasks such as creating 3D grids, plotting, generating random numbers, etc.
Thus, the user of HyVR should know a few of these tools. Most can be accomplished with numpy or scipy.

The current HyVR fork focuses on the implementation of geobodies on existing 3D grids.
A minimal example to assign values to a grid is exemplified below:

..code-block:: python

    #creating a grid with numpy according to the MODFLOW convention  
    
    xs = np.linspace(0,100,200)

    ys = np.linspace(0,80,160) . 
    
    zs = np.linspace(0,20,50) . 
    
    z,y,x = np.meshgrid(zs,ys,xs,indexing='ij') . 
    
    z = np.flip(z,axis=0)
    
    y = np.flip(y,axis=1)
    
    #defining the trough dimensions:
    
    centercoord = np.array([50,40,19])
    
    dims = np.array([40,20,9])
    
    #assigning values to the grid with the trough object:
    
    facies, dip, dip_dir = trough(x,y,z,centercoord,
                                                    dims,
                                                    azim = 20.,facies = np.array([1,2,3]),
                                                    internal_layering=True,
                                                    alternating_facies=True,
                                                    bulb=True,
                                                    layer_dist=3,
                                                    )

This simple code snippet can be made more complex for example simulating different 
Check the examples for comprehensible use of the package.
