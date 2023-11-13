"""
This script is used to run isp_pipeline.py on a dataset placed in ./inframes/normal/data
It also fetches if a separate config of a raw image is present othewise uses the default config
"""

import os
from pathlib import Path
from tqdm import tqdm
from infinite_isp import InfiniteISP

from util.config_utils import parse_file_name, extract_raw_metadata

DATASET_PATH = "./in_frames/normal/data/"
CONFIG_PATH = "./config/configs.yml"
VIDEO_MODE = False
EXTRACT_SENSOR_INFO = True
UPDATE_BLC_WB = True


def video_processing():
    """
    Processed Images in a folder [DATASET_PATH] like frames of an Image.
    - All images are processed with same config file located at CONFIG_PATH
    - 3A Stats calculated on a frame are applied on the next frame
    """

    raw_files = [f_name for f_name in os.listdir(DATASET_PATH) if ".raw" in f_name]
    raw_files.sort()

    infinite_isp = InfiniteISP(DATASET_PATH, CONFIG_PATH)

    # set generate_tv flag to false
    infinite_isp.c_yaml["platform"]["generate_tv"] = False
    infinite_isp.c_yaml["platform"]["render_3a"] = False

    for file in tqdm(raw_files, disable=False, leave=True):

        infinite_isp.execute(file)
        infinite_isp.load_3a_statistics()


def dataset_processing():
    """
    Processed each image as a single entity that may or may not have its config
    - If config file in the dataset folder has format filename-configs.yml it will
    be use to proocess the image otherwise default config is used.
    - For 3a-rendered output - set 3a_render flag in config file to true.
    """

    # The path for default config
    default_config = CONFIG_PATH

    # Get the list of all files in the DATASET_PATH
    directory_content = os.listdir(DATASET_PATH)

    # Get the list of all raw images in the DATASET_PATH
    raw_images = [
        x
        for x in directory_content
        if (Path(DATASET_PATH, x).suffix in [".raw", ".NEF", ".dng", ".nef"])
    ]

    infinite_isp = InfiniteISP(DATASET_PATH, default_config)

    # set generate_tv flag to false
    infinite_isp.c_yaml["platform"]["generate_tv"] = False

    is_default_config = True

    for raw in tqdm(raw_images, ncols=100, leave=True):

        raw_path_object = Path(raw)
        config_file = raw_path_object.stem + "-configs.yml"

        # check if the config file exists in the DATASET_PATH
        if find_files(config_file, DATASET_PATH):

            print(f"Found {config_file}.")

            # use raw config file in dataset
            infinite_isp.load_config(DATASET_PATH + config_file)
            is_default_config = False
            infinite_isp.execute()

        else:
            print(f"Not Found {config_file}, Changing filename in default config file.")

            # copy default config file
            if not is_default_config:
                infinite_isp.load_config(default_config)
                is_default_config = True

            if EXTRACT_SENSOR_INFO:
                if raw_path_object.suffix == ".raw":
                    print(
                        raw_path_object.suffix
                        + " file, sensor_info will be extracted from filename."
                    )
                    sensor_info = parse_file_name(raw)
                    if sensor_info:
                        infinite_isp.update_sensor_info(sensor_info)
                        print("updated sensor_info into config")
                    else:
                        print("No information in filename - sensor_info not updated")
                else:
                    sensor_info = extract_raw_metadata(DATASET_PATH + raw)
                    if sensor_info:
                        infinite_isp.update_sensor_info(sensor_info, UPDATE_BLC_WB)
                        print("updated sensor_info into config")
                    else:
                        print(
                            "Not compatible file for metadata - sensor_info not updated"
                        )

            infinite_isp.execute(raw)


def find_files(filename, search_path):
    """
    This function is used to find the files in the search_path
    """
    for _, _, files in os.walk(search_path):
        if filename in files:
            return True
    return False


if __name__ == "__main__":

    if VIDEO_MODE:
        print("PROCESSING VIDEO FRAMES ONE BY ONE IN SEQUENCE")
        video_processing()

    else:
        print("PROCESSING DATSET IMAGES ONE BY ONE")
        dataset_processing()
