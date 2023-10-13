
# Infinite-ISP
Infinite-ISP is a one stop solution for all your ISP development needs - from algorithms to an FPGA prototype and associated firmware, tools, etc. Its primary goal is to offer a unified platform that empowers ISP developers to accelerate ISP innovation. It includes a complete collection of camera pipeline modules written in Python, an FPGA bit-stream & the associated firmware for the implementation of the pipeline on the Kria KV260 development board and lastly a stand-alone Python based Tuning tool application for the pipeline.  The main components of the Infinite-ISP project are listed below:

| Repository name        | Description      | 
| -------------  | ------------- |
| **[Infinite-ISP_AlgorithmDesign](https://github.com/xx-isp/infinite-isp)** :anchor:                    | Python based model of the Infinite-ISP pipeline for algorithm development |
| **[Infinite-ISP_ReferenceModel](https://github.com/10xEngineersTech/Infinite-ISP_ReferenceModel)**                      | Python based fixed-point model of the Infinite-ISP pipeline for hardware implementation |
| **[Infinite-ISP_FPGABinaries](https://github.com/10xEngineersTech/Infinite-ISP_FPGABitstream)**                                      |FPGA binaries (bitstream + firmware executable) for the Xilinx® Kria KV260’s XCK26 Zynq UltraScale + MPSoC|
| **[Infinite-ISP_Firmware](https://github.com/10xEngineersTech/Infinite-ISP_Firmware)**                                      | Firmware for the Kria kV260’s embedded Arm® Cortex®A53 processor|
| **[Infinite-ISP_TuningTool](https://github.com/10xEngineersTech/Infinite-ISP_TuningTool)**                              | Collection of calibration and analysis tools for the Infinite-ISP |


# Infinite-ISP Algorithm Design: A Python-based Model for ISP Algorithm Development
Infinite-ISP Algorithm Design is a collections of camera pipeline modules implemented at the application level for converting an input RAW image from a sensor to an output RGB image. Infinite-isp aims to contain simple to complex algorithms at each modular level.


![](assets/infinite-isp-architecture-initial.png)

ISP pipeline for `infinite-isp v1.0`

## Objectives
Many open-source ISPs are available over the internet. Most of them are developed by individual contributors, each having its own strengths. This project aims to centralize all the open-source ISP development to a single place enabling all the ISP developers to have a single platform to contribute. Infinite-isp will not only contain the conventional algorithms but aims to contain state-of-the-art deep learning algorithms as well enabling a clean comparison between the two. This project has no bounds to ideas and is aimed to contain any algorithm that improves the overall results of the pipeline regardless of their complexity.

## Feature Comparison Matrix

A comparison of features with the famous openISP. 

Infinite-isp also tries to simulate the **3A-Algorithms**.

| Modules        | infinite-isp  | openISP        | 
| -------------  | ------------- |  ------------- |          
| Crop                                          | Bayer pattern safe cropping    | ---- |
| Dead Pixel Correction                         | Modified  [Yongji's et al, Dynamic Defective Pixel Correction for Image Sensor](https://ieeexplore.ieee.org/document/9194921) | Yes |
| Black Level Correction                        | Calibration / sensor dependent <br> - Applies BLC from config   | Yes |
| Optical Electronic Transfer Function (OECF)   | Calibration / sensor dependent <br> - Implements a LUT from config | ---- |
| Anti Aliasing Filter                          | ----  | Yes |
| Digital Gain                                  | Gains from config file | Brightness contrast control |
| Lens Shading Correction                       | To Be Implemented  | ---- |
| Bayer Noise Reduction                         | [Green Channel Guiding Denoising by Tan et al](https://www.researchgate.net/publication/261753644_Green_Channel_Guiding_Denoising_on_Bayer_Image)  | Chroma noise filtering |
| White Balance                                 | WB gains from config file  | Yes |
| CFA Interpolation                             | [Malwar He Cutler’s](https://www.ipol.im/pub/art/2011/g_mhcd/article.pdf ) demosaicing algo  | Yes <br> - Malvar He Cutler|
| **3A - Algorithms**                           | **AE & AWB** | ---- |
| Auto White Balance                            | - Gray World <br> - [Norm 2](https://library.imaging.org/admin/apis/public/api/ist/website/downloadArticle/cic/12/1/art00008) <br> - [PCA algorithm](https://opg.optica.org/josaa/viewmedia.cfm?uri=josaa-31-5-1049&seq=0) | ---- |
| Auto Exposure                                 | - [Auto Exposure](https://www.atlantis-press.com/article/25875811.pdf) based on skewness | ---- |
| Color Correction Matrix                       | Calibration / sensor dependent <br> - 3x3 CCM from config  | Yes <br> - 4x3 CCM  |
| Gamma Tone Mapping                            | Gamma LUT in RGB from config file  | Yes <br> - YUV and RGB domain|
| Color Space Conversion                        | YUV analogue and YCbCr digital <br> - BT 601 <br> - Bt 709  <br>   | Yes <br> - YUV analogue |
| Contrast Enhancement                          | Modified [contrast limited adaptive histogram equalization](https://arxiv.org/ftp/arxiv/papers/2108/2108.12818.pdf#:~:text=The%20technique%20to%20equalize%20the,a%20linear%20trend%20(CDF))  | ---- |
| Edge Enhancement / Sharpeining                | ---- | Yes | 
| Noise Reduction                               | [Non-local means filter](https://www.ipol.im/pub/art/2011/bcm_nlm/article.pdf)  | Yes <br> - NLM filter <br> - Bilateral noise filter|
| Hue Saturation Control                        | ---- | Yes |
| Scale                                         | - Integer Scaling  <br> - Non-Integer Scaling | ---- |
| False Color Suppression                       | ---- | Yes |      
| YUV Format                                    | - YUV - 444 <br> - YUV - 422 <br>  | ---- |


<!--- Commenting the checklist out and module list

## Modules
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

- [x] Crop
- [x] Dead Pixel Correction
- [ ] HDR Stitching
- [ ] Anti- Aliasing Filter
- [x] Black Level Calibration
- [x] Opto-electronic Trasnfer Function
- [x] Digital Gain
- [ ] Lens Shading Correction
- [x] Bayer Noise Reduction
- [x] White balance
- [x] CFA Interpolation
- [x] Auto White Balance
    - [x] Gray World
    - [x] Norm 2
    - [x] PCA
- [ ] Pre Gamma 
- [x] Color Correction Matrix
- [x] Gamma Tone Mapping
- [ ] Auto Exposure
    - [x] AE based on Skewness   
- [x] Color Space Conversion
- [x] Contrast Enchancement
- [x] 2d Noise Reduction
- [ ] Edge Enchancement / Sharpening
- [x] Scale
- [x] YUV Format
    - [x] 444
    - [x] 422 
- [ ] Compression  

## Usage
See a well descriptive [user guide](assets/User%20Guide.md). --->

## Dependencies
The project is compatible with `Python_3.9.12`

The dependencies are listed in the [requirements.txt](requirements.txt) file. 

The project assumes pip package manager as a pre-requisite.

## How to Run
Follow the following steps to run the pipeline
1.  Clone the repo using 

`git clone https://github.com/xx-isp/infinite-isp`

2.  Install all the requirements from the requirements file by running

`pip install -r requirements.txt`



## Example
There are a few sample images with tuned configurations already added to the project at [in_frames/normal](in_frames/normal) folder. In order to run any of these, just replace the config file name with any one of the sample configurations provided. For example to run the pipeline on `Indoor1_2592x1536_12bit_RGGB.raw` simply replace the config file name in `isp_pipeline.py` 

`config_path = './config/Indoor1-configs.yml'`

## Results
Here are the results of this pipeline compared with a market competitve ISP. 
The outputs of our ISP are displayed on the right, with the underlying ground truths on the left.

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; **ground truths**     &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; **infinite-isp** 
![](assets/Indoor1.png)
![](assets/Outdoor1.png)
![](assets/Outdoor2.png)
![](assets/Outdoor3.png)
![](assets/Outdoor4.png)

A comparison of the above results based on PSNR and SSIM image quality metrics

| Images    | PSNR  | SSIM  |
|-----------|-------|-------|
| Indoor1   | 21.51 | 0.8624 |
| Outdoor1  | 22.87 | 0.9431 |
| Outdoor2  | 20.54 | 0.8283 |
| Outdoor3  | 19.22 | 0.7867 |
| Outdoor4  | 22.25 | 0.8945 |

## User Guide

You can run the project by simply executing the [isp_pipeline.py](isp_pipeline.py). This is the main file that loads all the algorithic parameters from the [configs.yml](config/configs.yml)
The config file contains tags for each module implemented in the pipeline. A brief description as well as usage of each module is as follows:

### Platform

| platform            | Details | 
| -----------         | --- |
| filename            | Specifies the file name for running the pipeline. The file should be placed in the [in_frames/normal](in_frames/normal) directory 
| disable_progress_bar| Enables or disables the progress bar for time taking modules
| leave_pbar_string   |  Hides or unhides the progress bar upon completion

### Sensor_info

| sensor Info   | Details | 
| -----------   | --- |
| bayer_pattern | Specifies the bayer patter of the RAW image in lowercase letters <br> - `bggr` <br> - `rgbg` <br> - `rggb` <br> - `grbg`|
| range         | Not used |
| bitdep        | The bit depth of the raw image |
| width         | The width of the input raw image |
| height        | The height of the input raw image |
| hdr           | Not used |

### Crop

| crop          | Details | 
| -----------   | --- |
| isEnable      |  Enables or disables this module. When enabled it ony crops if bayer pattern is not disturbed
| isDebug       |  Flag to output module debug logs 
| new_width     |  New width of the input RAW image after cropping
| new_height    |  New height of the input RAW image after cropping

### Dead Pixel Correction 

| dead_pixel_correction | Details |
| -----------           |   ---   |
| isEnable              |  Enables or disables this module
| isDebug               |  Flag to output module debug logs 
| dp_threshold          |  The threshold for tuning the dpc module. The lower the threshold more are the chances of pixels being detected as dead and hence corrected  

### HDR Stitching 

To be implemented

### Black Level Correction 

| black_level_correction  | Details |
| -----------             |   ---   |
| isEnable                |  Enables or disables this module
| r_offset                |  Red channel offset
| gr_offset               |  Gr channel offset
| gb_offset               |  Gb channel offset
| b_offset                |  Blue channel offset
| isLinear                |  Enables or disables linearization. When enabled the BLC offset maps to zero and saturation maps to the highest possible bit range given by  the user  
| r_sat                   | Red channel saturation level  
| gr_sat                  |  Gr channel saturation level
| gb_sat                  |  Gb channel saturation level
| b_sat                   |  Blue channel saturation level

### Opto-Electronic Conversion Function 

| OECF  | Details |
| -----------     |   ---   |
| isEnable        | Enables or disables this module
| r_lut           | The look up table for oecf curve. This curve is mostly sensor dependent and is found by calibration using some standard technique 

### Digital Gain

| digital_gain    | Details |
| -----------     |   ---   |
| isEnable        | This is a essential module and cannot be disabled 
| isDebug         | Flag to output module debug logs
| gain_array      | Gains array. User can select any one of the gain listed here. This module works together with AE module  |
| current_gain    | Index for the current gain starting from zero |

### Lens Shading Calibration 

To be implemented

### Bayer Noise Reduction

| bayer_noise_reduction   | Details |
| -----------             |   ---   |
| isEnable                | When enabled reduces the noise in bayer domain using the user given parameters |
| filt_window             | Should be an odd window size
| r_stdDevS               | Red channel gaussian kernel strength. The more the strength the more the blurring. Cannot be zero  
| r_stdDevR               | Red channel range kernel strength. The more the strength the more the edges are preserved. Cannot be zero
| g_stdDevS               | Gr and Gb gaussian kernel strength
| g_stdDevR               | Gr and Gb range kernel strength
| b_stdDevS               | Blue channel gaussian kernel strength
| b_stdDevR               | Blue channel range kernel strength


### White balance

| white_balance           | Details |
| -----------             |   ---   |
| isEnable                | Applies user given white balance gains when enabled |
| isAuto                  | When true enables the 3A - AWB and does'nt use the user given WB gains |
| r_gain                  | Red channel gain  |
| b_gain                  | Blue channel gain |

### CFA Interpolation (Demosaicing)

| demosaic                | Details |
| -----------             |   ---   |
| isEnable                | This is a essential module and cannot be disabled |

### 3A - Auto White Balance (AWB)
| auto_white_balance      | Details |
| -----------             |   ---   |
| algorithm               | Can select one of the following algos <br> - `grey_world`  <br> - `norm_2`  <br> - `pca` |
| percentage              | [0 - 100] - Parameter to select dark-light pixels percentage for pca algorithm |

### Color Correction Matrix (CCM)

| color_correction_matrix                 | Details |
| -----------                             |   ---   |
| isEnable                                | When enabled applies the user given 3x3 CCM to the 3D RGB image having rows sum to 1 convention  |
| corrected_red                           | Row 1 of CCM
| corrected_green                         | Row 2 of CCM
| corrected_blue                          | Row 3 of CCM

### Gamma Correction
| gamma_correction        | Details |
| -----------             |   ---   |
| isEnable                | When enabled  applies tone mapping gamma using the LUT  |
| gammaLut                | The look up table for gamma curve |

### 3A - Auto Exposure
| auto_exposure      | Details                                                                                      
|--------------------|----------------------------------------------------------------------------------------------|
| isEnable           | When enabled applies the 3A- Auto Exposure algorithm                                         |
| isDebug            | Flag to output module debug logs                                                             |  
| center_illuminance | The value of center illuminance for skewness calculation ranges from 0 to 255. Default is 90 |   
| histogram_skewness | The range of histogram skewness should be between 0 and 1 for correct exposure calculation   |  

### Color Space Conversion (CSC)

| color_space_conversion | Details                                                                             |  
|------------------------|-------------------------------------------------------------------------------------|
| isEnable               | This is a essential module and cannot be disabled                                   |   
| conv_standard          | The standard to be used for conversion <br> - `1` : Bt.709 HD <br> - `2` : Bt.601/407 |   
| conv_type              | The conversion type <br> - `1` : Analogue YUV <br> - `2` : Digital YCbCr               |      

### Contrast Enchancement 

| ldci       | Details                                                                      | 
|------------|----------------------------------------------------------------------------- |
| isEnable   | When enabled local dynamic contrast enhancement is applied to the Y channel  |  
| clip_limit | The clipping limit that controls amount of detail to be enhanced             |   
| wind       | Window size for applying filter                                              |   

### Edge Enchancement / Sharpening 

To be implemented

### 2d Noise Reduction

| 2d_noise_reduction | Details                                           | 
|--------------------|---------------------------------------------------|
| isEnable           | When enabled applies the non-local mean filtering |   
| window_size        | Search window size for applying the filter        |   
| patch_size         | Patch window size for applying filter             |  
| h                  | Strength of blurring                              | 

### Scaling 

| scale            | Details |   
|------------------|---------------------------------------------------------------------------------------------------------------------------------------------------
| isEnable         | When enabled down scales the input image                                                                                                 
| isDebug          | Flag to output module debug logs                                                                                                           
| new_width        | Down scaled width of the output image                                                                                                      
| new_height       | Down scaled height of the output image                                                                                                       
| isHardware       | When true applies the hardware friendly techniques for downscaling. This can only be applied to any one of the input sizes 3 input sizes and can downscale to <br> - `2592x1944` to `1920x1080` or `1280x960` or `1280x720` or `640x480` or `640x360`  <br> - `2592x1536` to `1280x720` or `640x480` or `640x360` <br> - `1920x1080` to to `1280x720` or `640x480` or `640x360`  |  
| Algo             | Software friendly scaling. Only used when isHardware is disabled <br> - `Nearest_Neighbor` <br> - `Bilinear`                                       
| upscale_method   | Used only when isHardware enabled. Upscaling method, can be one of the above algos                                                          
| downscale_method | Used only when isHardware enabled. Downscaling method, can be one of the above algos 

### YUV Format 
| yuv_conversion_format     | Details                                                |
|---------------------------|--------------------------------------------------------|
| isEnable                  | Enables or disables this module                        |   
| conv_type                 | Can convert the YCbCr to YUV <br> - `444` <br> - `422` |  

### Pre-Gamma
TBD
### Tone-Mapping
TBD
### Jpeg-Compression
TBD

## FAQ
**Why is it named infinite-isp?**

ISPs are hardware dependent. In them algorithms are limited to perform to their best because of hardware limitations. Infinite-isp tends to somewhat remove this limitation and let the algorithms perform to the full potential targeting best results. 

**Will inifnite-isp also contain algorithms that involve machine learning?**

Yes definitely this is mainly because it is seen that machine learning models tend to give perform much better results as compared to conventional models. The plan is as follows

- The release `v0.x` till `v1.0` will involve buildng a basic ISP pipelne at conventional level. 

- The release `v1.0` will have all camera pipeline modules implemented at conventional level. **This release will mostly contain algorithms that can be easily ported to hardware ISPs** 

- `v1.x.x` releases will have all the necessary improvements of these conventional algorithms till release `v2.0`

- From release `v2.0` infinite-isp will start implementing machine learning models for specific algorithms. 

- Release `v3.0` will have infinite-isp having both conventional and deep learning algorithms (not for all pipeline modules but for specific ones)

## License 
This project is licensed under Apache 2.0 (see [LICENSE](LICENSE) file).

## Acknowledgments
- This project started of from the inspiration of [cruxopen/openISP](https://github.com/cruxopen/openISP.git)

## List of Open Source ISPs
- [openISP](https://github.com/cruxopen/openISP.git)
- [Fast Open Image Signal Processor](https://github.com/QiuJueqin/fast-openISP.git)
- [AbdoKamel - simple-camera-pipeline](https://github.com/AbdoKamel/simple-camera-pipeline.git)
- [Mushfiqulalam - isp](https://github.com/mushfiqulalam/isp)
- [Karaimer - A Software Platform for Manipulating the Camera Imaging Pipeline](https://karaimer.github.io/camera-pipeline)
- [rawpy](https://github.com/letmaik/rawpy.git)

## Contact
For any inquiries or feedback feel free to reach out.

Email: isp@10xengineers.ai

Website: http://www.10xEngineers.ai

LinkedIn: https://www.linkedin.com/company/10x-engineers/
