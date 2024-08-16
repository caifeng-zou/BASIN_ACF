# BASIN_ACF

## Introduction
This repository provides code for the paper [Imaging the Northern Los Angeles Basins with Autocorrelations](https://arxiv.org/pdf/2405.20635).

## File Description
- **noise_acf_TT_SB1.ipynb**: Workflow for ambient noise autocorrelation, taking Line SB1 as an example
- **earthquake_acf_TT_SB1.ipynb**: Workflow for teleseismic wave autocorrelation, taking Line SB1 as an example
- **noise_func.py**: Functions for use with noise_acf_TT_SB1.ipynb
- **earthquake_func.py**: Functions for use with earthquake_acf_TT_SB1.ipynb
- ***.npy**: Final autocorrelation functions
- ***.png**: Example images


## Dependencies
```
environment.yml
```

## Example Image
![image](noise_example.png)
![image](teleseismic_example.png)

## Contact
We welcome any comments or questions regarding this work. If you find it helpful, please cite:
```
Zou, C., & Clayton, R. W. (2024). Imaging the Northern Los Angeles Basins with Autocorrelations. arXiv preprint arXiv:2405.20635.
```

Caifeng Zou\
czou@caltech.edu

