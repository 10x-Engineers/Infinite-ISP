# infinite-isp

## Overview
Infinite-isp is a collections of camera pipeline modules implemented at the application level for converting an input RAW image from an sensor to an output RGB image. Infinite-isp aims to contain simple to complex algorithms at each modular level.

## Objectives
The project aims to produce outputs that are fine tuned to best possible results with the algorithms implemented at each module level. The project aims to have both conventional and deep learning state-of-the-art algorithms implemented in a single project enabling a clean comparison between the two. The `infinite` in the name is just to depict that this project has no bounds to ideas and is aimed to contain any algorithm that improves the overall results of the pipeline regardless of their complexity.


## Modules
![](assets/infinite-isp-architecture-initial.png)

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

Since the color correction matrix is tuned offline and is stored in the ISP as register parameters the current module implements the application of color correction matrix provided in the `config.yml` file. 

### Color Space Conversion

Algorithm details goes here

### 2D Noise Reduction

Algorithm details goes here

### Sharpen

Algorithm details goes here

### JPEG Conversion

Algorithm details goes here

## Algorithm Implementation Checklist

- [ ] Dead Pixel Correction
- [ ] HDR Stitching
- [ ] Lens Shading Correction
- [ ] Bayer Noise Reduction
- [ ] Black Level Calibration
- [x] Auto White Balance
    - [x] Gray World
- [ ] Pre Gamma 
- [ ] Tone Mapping
- [x] Demosaicing

- [x] Color Correction Matrix
- [ ] Color Space Conversion
- [ ] 2d Noise Reduction
- [ ] Sharpening
- [ ] Compression / Conversion

## Usage
Each pipeline module can be tuned from the `configs.yml` configuration file . The configuration file contains all the modules. Each module contains algorithms and algorithm specific parameters.  

## Dependencies
The project is written in python3 which requires the following packages to be installed before running.  
- `matplotlib`
- `numpy`
- `yaml`

## FAQ
**Why is it named infinite-isp?**

ISPs are hardware dependent. In them algorithms are limited to perform to their best because of hardware limitations. Infinite-isp tends to somewhat remove this limitation and let the algorithms perform to the full potential targeting best results. 

**Will inifnite-isp also contain algorithms that involve machine learning?**

Yes definitely this is mainly because it is seen that machine learning models tend to perform much better results as compared to conventional models. The plan is as follows

- The release `v0.x` till `v1.0` will involve buildng a basic ISP pipelne at conventional level. 

- The release `v1.0` will have all camera pipeline modules implemented at conventional level. `v1.x.x` releases will have all the necessary improvements of these conventional algorithms till release `v2.0`

- From release `v2.0` infinite-isp will start implementing machine learning models for specific algorithms. 

- Release `v3.0` will have infinite-isp having both conventional and deep learning algorithms (not for all pipeline modules but for specific ones)

## License 
MIT License (see [LICENSE](LICENSE) file).

## Acknowledgments
- This project started of from the inspiration of [cruxopen/openISP](https://github.com/cruxopen/openISP.git)
