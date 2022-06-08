from matplotlib import pyplot as plt
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime
from modules.dead_pixel_correction import DeadPixelCorrection as DPC
from modules.hdr_stitching import HdrStitching as HDRS
from modules.lens_shading_correction import LensShadingCorrection as LSC
from modules.bayer_noise_reduction import BayerNoiseReduction as BNR
from modules.black_level_correction import BlackLevelCorrection as BLC
from modules.white_balance import WhiteBalance as WB
from modules.gamma_correction import GammaCorrection as GC
from modules.tone_mapping import ToneMapping as TM
from modules.demosaic import CFAInterpolation as CFA_I
from modules.color_correction_matrix import ColorCorrectionMatrix as CCM
from modules.color_space_conversion import ColorSpaceConv as CSC
from modules.noise_reduction_2d import NoiseReduction2d as NR2D
from modules.sharpen import Sharpening as SHARP


raw_path = './in_frames/normal/ColorCheckerRAW_1920x1080_12bit_BGGR.RAW'
config_path = './config/configs.yml'
inFile = Path(raw_path).stem
outFile = "Out_" + inFile

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
parm_lsc = c_yaml['lens_shading_correction']
parm_bnr = c_yaml['bayer_noise_reduction']
parm_blc = c_yaml['black_level_correction']
parm_wbc = c_yaml['white_balance']
parm_gmm = c_yaml['pre_gamma']
parm_tmp = c_yaml['tone_mapping']
parm_dem = c_yaml['demosaic']
parm_ccm = c_yaml['color_correction_matrix']
parm_gmc = c_yaml['gamma_correction']
parm_csc = c_yaml['color_space_conversion']
parm_2dn = c_yaml['2d_noise_reduction']
parm_sha = c_yaml['sharpen']
parm_jpg = c_yaml['jpeg_conversion']

# Load Raw
raw = np.fromfile(raw_path, dtype=np.uint16).reshape((height, width))
# raw = np.float32(raw) / np.power(2,16)

print(50*'-' + '\nLoading RAW Image Done......\n')

# Dead pixels correction (1)
dpc = DPC(raw, sensor_info, parm_dpc)
dpc_raw = dpc.execute()

# HDR stitching (2)
hdr_st = HDRS(dpc_raw, sensor_info, parm_hdr)
hdr_raw = hdr_st.execute()

# Lens shading correction (3)
lsc = LSC(hdr_raw, sensor_info, parm_lsc)
lsc_raw = lsc.execute()

# Bayer noise reduction (4)
bnr = BNR(lsc_raw, sensor_info, parm_bnr)
bnr_raw = bnr.execute()

# Black level correction (5)
blc = BLC(bnr_raw, sensor_info, parm_blc)
blc_raw = blc.execute()

# White balancing (6)
wb = WB(blc_raw, sensor_info, parm_wbc)
wb_raw = wb.execute()

# Gamma (7)
gc = GC(wb_raw, sensor_info, parm_gmm)
gamma_raw = gc.execute()

# Tone mapping (8)
tmap = TM(gamma_raw, sensor_info, parm_tmp)
tmap_img = tmap.execute()

# CFA demosaicing (9)
cfa_inter = CFA_I(tmap_img, sensor_info, parm_dem)
demos_img = cfa_inter.execute()

# Color correction matrix (10)
ccm = CCM(demos_img, sensor_info, parm_ccm)
ccm_img = ccm.execute()

# Color space conversion
csc = CSC(ccm_img, sensor_info, parm_csc)
csc_img = csc.execute()

# 2d noise reduction
nr2d = NR2D(csc_img, sensor_info, parm_csc)
nr2d_img = nr2d.execute()

# Sharpening
sharp = SHARP(nr2d_img, sensor_info, parm_sha)
sharp_img = sharp.execute()


# plt.imshow(sharp_img)
# plt.show()

print(50*'-' + '\n')
dt_string = datetime.now().strftime("_%Y%m%d_%H%M%S")
plt.imsave("./out_frames/" + outFile + dt_string + ".png", sharp_img)
