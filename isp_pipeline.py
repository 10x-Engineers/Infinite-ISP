from matplotlib import pyplot as plt
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime
from modules.dead_pixel_correction import DeadPixelCorrection as DPC
from modules.hdr_stitching import HdrStitching as HDRS
from modules.digital_gain import DigitalGain as DG
from modules.lens_shading_correction import LensShadingCorrection as LSC
from modules.bayer_noise_reduction import BayerNoiseReduction as BNR
from modules.black_level_correction import BlackLevelCorrection as BLC
from modules.oecf import OECF
from modules.white_balance import WhiteBalance as WB
from modules.auto_white_balance import AutoWhiteBalance as AWB
from modules.gamma_correction import GammaCorrection as GC
from modules.tone_mapping import ToneMapping as TM
from modules.demosaic import CFAInterpolation as CFA_I
from modules.color_correction_matrix import ColorCorrectionMatrix as CCM
from modules.color_space_conversion import ColorSpaceConv as CSC
from modules.ldci import LDCI as LDCI
from modules.yuv_conv_format import YUVConvFormat as YUV_C
from modules.noise_reduction_2d import NoiseReduction2d as NR2D
from modules.sharpen import Sharpening as SHARP


raw_path = './in_frames/normal/ColorChecker_2592x1536_12bits_RGGB.raw'
config_path = './config/configs.yml'
inFile = Path(raw_path).stem
outFile = "Out_" + inFile

#not to jumble any tags
yaml.preserve_quotes = True

with open(config_path, 'r') as f:
    c_yaml = yaml.safe_load(f)

# Extract basic sensor info
sensor_info = c_yaml['sensor_info']
range = sensor_info['range']
bayer = sensor_info['bayer_pattern']
width = sensor_info['width']
height = sensor_info['height']
bpp = sensor_info['bitdep']


# Get isp module params
parm_dpc = c_yaml['dead_pixel_correction']
parm_hdr = c_yaml['hdr_stitching']
parm_dga = c_yaml['digital_gain']
parm_lsc = c_yaml['lens_shading_correction']
parm_bnr = c_yaml['bayer_noise_reduction']
parm_blc = c_yaml['black_level_correction']
parm_oec = c_yaml['OECF']
parm_wbc = c_yaml['white_balance']
parm_awb = c_yaml['auto_white_balance']
parm_gmm = c_yaml['pre_gamma']
parm_tmp = c_yaml['tone_mapping']
parm_dem = c_yaml['demosaic']
parm_ccm = c_yaml['color_correction_matrix']
parm_gmc = c_yaml['gamma_correction']
parm_csc = c_yaml['color_space_conversion']
parm_ldci = c_yaml['ldci']
parm_2dn = c_yaml['2d_noise_reduction']
parm_sha = c_yaml['sharpen']
parm_jpg = c_yaml['jpeg_conversion']
parm_yuv = c_yaml['yuv_conversion_format']

# Load Raw
raw = np.fromfile(raw_path, dtype=np.uint16).reshape((height, width))
# raw = np.float32(raw) / np.power(2,16)

print(50*'-' + '\nLoading RAW Image Done......\n')

# 1 Dead pixels correction
dpc = DPC(raw, sensor_info, parm_dpc)
dpc_raw = dpc.execute()

# 2 HDR stitching
# TODO figure out where in pipeline HDR stitching will happen and fix this numbering accordingly
hdr_st = HDRS(dpc_raw, sensor_info, parm_hdr)
hdr_raw = hdr_st.execute()

# 3 Black level correction
blc = BLC(hdr_raw, sensor_info, parm_blc)
blc_raw = blc.execute()

# 4 OECF
oecf = OECF(blc_raw, sensor_info, parm_oec)
oecf_raw = oecf.execute()

# 5 Digital Gain
dga = DG(oecf_raw, sensor_info, parm_dga)
dga_raw = dga.execute()

# 6 Lens shading correction
lsc = LSC(dga_raw, sensor_info, parm_lsc)
lsc_raw = lsc.execute()

# 7 Bayer noise reduction
bnr = BNR(lsc_raw, sensor_info, parm_bnr)
bnr_raw = bnr.execute()

# 8 White balancing
wb = WB(bnr_raw, sensor_info, parm_wbc)
wb_raw = wb.execute()

# 9 CFA demosaicing
cfa_inter = CFA_I(wb_raw, sensor_info, parm_dem)
demos_img = cfa_inter.execute()

awb = AWB(demos_img, sensor_info, parm_wbc, parm_awb)
awb_img = awb.execute()

# 10 Color correction matrix
ccm = CCM(awb_img, sensor_info, parm_ccm)
ccm_img = ccm.execute()

# 11 Gamma
gc = GC(ccm_img, sensor_info, parm_gmc)
gamma_raw = gc.execute()

# 12 Color space conversion
csc = CSC(gamma_raw, sensor_info, parm_csc)
csc_img = csc.execute()

# Local Dynamic Contrast Improvement
ldci = LDCI(csc_img, sensor_info, parm_ldci)
ldci_img = ldci.execute()

# 14 Sharpening
sharp = SHARP(ldci_img, sensor_info, parm_sha)
sharp_img = sharp.execute()

# 15 2d noise reduction
nr2d = NR2D(sharp_img, sensor_info, parm_2dn)
nr2d_img = nr2d.execute()

# 16 YUV saving format 444, 422 etc
yuv = YUV_C(nr2d_img, sensor_info, parm_yuv, inFile, parm_csc)
yuv_conv = yuv.execute()

#only to view image if csc is off it does nothing
out_img = csc.yuv_to_rgb(nr2d_img)

# plt.imshow(sharp_img)
# plt.show()

print(50*'-' + '\n')
dt_string = datetime.now().strftime("_%Y%m%d_%H%M%S")

#save config
with open("./out_frames/" + outFile + dt_string+'.yaml', 'w') as file:
    yaml.dump(c_yaml, file, sort_keys= False)
#save image
plt.imsave("./out_frames/" + outFile + dt_string + ".png", out_img)