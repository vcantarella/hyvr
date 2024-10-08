# HyVR - simple

![Build Status](https://github.com/vcantarella/hyvr/actions/workflows/python-package-conda.yml/badge.svg?event=push)
[![codecov](https://codecov.io/github/vcantarella/hyvr/graph/badge.svg?token=QWGCQVEJ3G)](https://codecov.io/github/vcantarella/hyvr)

This is a fork from the original HyVR package with minimal implementation of the core features the idea to keep the original ideas alive, but maintainable in the fast python development ecosystem.

The Hydrogeological Virtual Reality simulation package (HyVR) is a Python module that helps researchers and practitioners generate sedimentary subsurface models with multiple scales of heterogeneity that are based on geological concepts. The simulation outputs can then be used to explore groundwater flow and solute transport behaviour, for example. The user must work with a previously created regular grid, which can be used directly in MODFLOW or interpolated to the simulation grid that the user will need.

The original motivation for HyVR was the lack of tools for modelling sedimentary deposits that include bedding structure model outputs (i.e., dip and azimuth).
Such bedding parameters were required to approximate full hydraulic-conductivity tensors for groundwater flow modelling. HyVR is able to simulate these bedding
parameters and generate spatially distributed parameter fields, including full hydraulic-conductivity tensors. A documentation on the simplified HyVR is being built in: *TO BE ADDED*.

For more information, the original HyVR information is available in the online `technical documentation <https://driftingtides.github.io/hyvr/index.html>`_.

For citation of the original software and development, please use the following reference:

* Bennett, J. P., Haslauer, C. P., Ross, M., & Cirpka, O. A. (2018). An open, object-based framework for generating anisotropy in sedimentary subsurface models. Groundwater. DOI:* [10.1111/gwat.12803](https://onlinelibrary.wiley.com/doi/abs/10.1111/gwat.12803>).
    * A preprint version of the article is available* [here](https://github.com/driftingtides/hyvr/blob/master/docs/Bennett_GW_2018.pdf) .

## Installing the HyVR package
--------------------------------------

The package should work whether you are using anaconda or another virtual environment.
Your working environment should have updated installations of the following libraries:

- numpy
- numba
- scipy

Once you have activated your virtual environment, you can install HyVR
using `pip`. First git clone the repository,
navigate to the root directory and type::

```    pip install .```


## Usage
-----

See the documentation and the examples folder for usage.

## Development
-----------

The orgininal HyVR has been developed by Jeremy Bennett ([website](https://jeremypaulbennett.weebly.com))
as part of his doctoral research at the University of Tübingen and by Samuel
Scherrer as a student assistant.

The current version is maintained by Vitor Cantarella ([website](https://github.com/vcantarella))

## Problems, Bugs, Unclear Documentation
-------------------------------------

please use the Issues page for bugs.
