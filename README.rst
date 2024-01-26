====================================================================
Introduction
====================================================================

**HyVR**

This is a fork from the original HyVR package with minimal implementation of
the core features the idea to keep the original ideas alive, 
but maintainable in the fast python development ecosystem.
Thus, most of the non-essencial functionality, that is nowadays done much better in other packages, 
has been dropped. The functionality is now focus on the creation of geobodies to grids.

Eveything that could be done before is still doable, and more. But the user should understand a bit more about
operating on arrays and using python.


The Hydrogeological Virtual Reality simulation package (HyVR) is a Python module
that helps researchers and practitioners generate sedimentary subsurface models with
multiple scales of heterogeneity that are based on geological concepts. The
simulation outputs can then be used to explore groundwater flow and solute
transport behaviour, for example. The user must work with a previously created regular grid, which
can be used directly in MODFLOW or interpolated to the simulation grid that the user will need.

The original motivation for HyVR was the lack of tools for modelling sedimentary
deposits that include bedding structure model outputs (i.e., dip and azimuth).
Such bedding parameters were required to approximate full hydraulic-conductivity
tensors for groundwater flow modelling. HyVR is able to simulate these bedding
parameters and generate spatially distributed parameter fields, including full
hydraulic-conductivity tensors. A documentation on the simplified HyVR is being built in: *TO BE ADDED*
The original HyVR information is available in the online `technical documentation <https://driftingtides.github.io/hyvr/index.html>`_.

For citation of the original software and development, please use the following reference:

*HyVR can be attributed by citing the following journal article: Bennett, J. P.,
Haslauer, C. P., Ross, M., & Cirpka, O. A. (2018). An open, object-based
framework for generating anisotropy in sedimentary subsurface
models. Groundwater.
DOI:* `10.1111/gwat.12803 <https://onlinelibrary.wiley.com/doi/abs/10.1111/gwat.12803>`_.
*A preprint version of the article is available* `here <https://github.com/driftingtides/hyvr/blob/master/docs/Bennett_GW_2018.pdf>`_.

Installing the HyVR package
--------------------------------------

Installing HyVR
^^^^^^^^^^^^^^^

The package should work whether you are using anaconda or another virtual environment.
Your working environment should have updated installations of the following libraries:

-numpy
-numba
-scipy

Once you have activated your virtual environment, you can install HyVR
using ``pip``. First git clone the repository,
navigate to the root directory and type::

    pip install .

Alternatively, you can use the package without installing it by typing in your script::

    sys.path.append("path where HyVR is") 

Usage
-----

See the documentation and the examples folder for usage.

Development
-----------

The orgininal HyVR has been developed by Jeremy Bennett (`website <https://jeremypaulbennett.weebly.com>`_)
as part of his doctoral research at the University of Tübingen and by Samuel
Scherrer as a student assistant.

The current version is maintained by Vitor Cantarella (`website <https://vcantarella.gitub.io>`_)

Problems, Bugs, Unclear Documentation
-------------------------------------

please use the Issues page for bugs.
