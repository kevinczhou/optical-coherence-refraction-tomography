# Optical coherence refraction tomography

This repository contains Python code that implements optical coherence refraction tomography (OCRT), a technique which starts with low-resolution optical coherence tomography (OCT) images acquired from multiple angles, and through iterative optimization generates simultaneously 1) a high-resolution reconstruction, and 2) a refractive index map of the sample. For more details, you can read our paper at https://www.nature.com/articles/s41566-019-0508-1 (or, if you don't have a subscription to Nature Photonics, https://rdcu.be/bO6eQ).

More recently, we have extended OCRT to spectroscopic OCT (SOCT). The new technique, termed spectroscopic OCRT (SOCRT), circumvents the trade-off between axial resolution and spectral resolution in SOCT, thus enabling reconstructions with simultaneously high spatial and spectral resolution (https://www.osapublishing.org/ol/abstract.cfm?uri=ol-45-7-2091).

See also our 3D implementation of OCRT, featuring a new hardware and a new, more efficient, GPU-accelerated reconstruction algorithm: https://github.com/kevinczhou/3d-ocrt

## Data
This code generates OCRT results similar to those in figures 4-6 of our paper, which feature 7 different biological samples:
- mouse_vas_deferens1
- mouse_vas_deferens2
- mouse_femoral_artery
- mouse_bladder
- mouse_trachea
- human_cornea
- insect_leg

These 7 datasets can be downloaded from [here](https://doi.org/10.6084/m9.figshare.8297138) as `.mat` files. They are 80-120 MB each.

## Code
The code depends on the following libraries:
- tensorflow (the CPU version is sufficient)
- numpy
- scipy
- opencv
- matplotlib
- jupyter

With these libraries installed and the datasets downloaded into the `data/` directory, you should be able to run the jupyter notebook as is.

I tested this code for all 7 datasets using Python 2.7 with TensorFlow 1.8 on a desktop running Ubuntu 16.04 with 48 GB of RAM. I expect that the code should work with later versions of TensorFlow (before 2.0) and in Python 3, though I did not test these as thoroughly as I did for Python 2/TensorFlow 1.8. Expect slightly different results.

Depending on the sample, this code could end up exceeding 40 GB of RAM usage, so I recommend using a machine with at least that much memory. With the default settings in the code, expect on the order of several hours to around a day (a few minutes per iteration) for the optimization loops to run. Also expect the saved TensorFlow graph to take up ~500 MB of disk space per sample.

As the authors have a currently-pending patent related to OCRT, you may only use this code for non-commercial purposes.

## Citation
If you find our code and/or datasets useful to your research, please cite the following publication:

Zhou, K. C., Qian, R., Degan, S., Farsiu, S., & Izatt, J. A. Optical coherence refraction tomography. *Nature Photonics*, 13(11), 794-802 (2019).



