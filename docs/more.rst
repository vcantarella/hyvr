====================================================================
Extending HyVR
====================================================================

The HyVR package is a work in progress. It has been implemented in Python in order to make it
accessible for researchers, easily customisable, and hopefully somewhat easy to extend to suit the
needs of future research.
Extending HyVR has been significantly simplified. We hope that this attracts more collaboration and keep the project alive.


------------------------------------------------------------------------
Adding more geometries
------------------------------------------------------------------------

HyVR has been set up in such a way to facilitate the implementation of additional hydrofacies
assemblage geometries.

In order to generate new types of geometries a new function representing a new geobody has to be added
(in ``objects/``).

Each function need to take in the grid coordinates (x,y,z) and some parameter, such as the facies.
Ideally, it should follow the same naming and typing from the original functions. The function must output the facies code, the dip and dip direction at the grid cells.
Moreover, the new geobody or internal heterogeneity should be jittable with numba @numba.njit.

When adding geometries, we suggest reviewing the code for the existing geometries to see how it is
currently implemented. This should provide a reasonable idea of how to put a new geometry together.

------------------------------------------------------------------------
The HyVR wish list
------------------------------------------------------------------------

Any modelling project will have 'areas for growth' (as opposed to weaknesses).

* Some level of conditioning, or improved interfacing with multiple-point geostatistical packages.
* Interaction of extruded parabolas, as well as more complex/realisitic configurations of channel deposits (e.g. point bars).
* Utilities for deriving HyVR simulation parameters from transitional probability geostatistics.
