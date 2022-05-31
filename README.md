# infinite-isp

## Overview
Infinite-isp is a collections of camera pipeline modules implemented at the application level for converting an input RAW image from an sensor to an output RGB image. Infinite-isp aims to contain simple to complex algorithms at each modular level.

## Objectives
The project aims to produce outputs that are fine tuned to best possible results with the algorithms implemented at each module level. The project aims to have both conventional and deep learning state-of-the-art algorithms implemented in a single project enabling a clean comparison between the two. The `infinite` in the name is just to depict that this project has no bounds to ideas and is aimed to contain any algorithm that improves the overall results of the pipeline regardless of their complexity.


## Modules
![](https://github.com/xx-isp/infinite-isp/blob/main/assets/infinite-isp-architecture-initial.png)

### Dead Pixel Correction

Algorithm details goes here 

### HDR Stitching

Algorithm details goes here

### Lens Shading Correction

Algorithm details goes here

### Bayer Noise Reduction

Algorithm details goes here

### Black Level Calibration

Algorithm details goes here

### White Balance

Implements the gray world white balancing algorithm in the bayer domain.

### Pre Gamma

Algorithm details goes here

### Tone Mapping

Algorithm details goes here

### Demosaic

Implements the bilinear iterpolation for cfa demosaicing

O. Losson, L. Macaire, and Y. Yang. Comparison of color demosaicing methods. _In Advances in Imaging and Electron Physics_, volume 162, pages 173–265. 2010. [ doi:10.1016/S1076-5670(10)62005-8](https://doi.org/10.1016/S1076-5670(10)62005-8)

### Color Correction Matrix

Since the color correction matrix is tuned offline and is stored in the isp as register parameters the current module implements the application of color correction matrix provided in the `config.yml` file. 

### Color Space Conversion

Algorithm details goes here

### 2D Noise Reduction

Algorithm details goes here

### Sharpen

Algorithm details goes here

### JPEG Conversion

Algorithm details goes here

## Usage
Each pipeline module can be tuned from the `configs.yml` configuration file . The configuration file contains all the modules. Each module contains algorithms and algorithm specific parameters.  

