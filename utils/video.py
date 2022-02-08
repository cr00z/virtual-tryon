# Save/load functions for video processing
# https://github.com/cr00z/virtual_tryon


import cv2
import os.path
from typing import Tuple

import numpy as np


def get_video(input_path: str) -> Tuple[cv2.VideoCapture, int]:
    """
    Get video capture object from video file
    :param input_path: path to video file
    :return: VideoCapture object, num of frames
    """
    cap = cv2.VideoCapture(input_path)
    assert cap.isOpened(), f'Failed to opening video: {input_path}'
    cap_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, cap_length


def write_image(out_path: str, current_frame: int, image: np.ndarray) -> None:
    target_dir = os.path.join(out_path, "frames")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    fpath = os.path.join(target_dir, f"{current_frame:05d}.jpg")
    cv2.imwrite(fpath, image)


def generate_video(workdir: str) -> None:
    in_dir = os.path.join(workdir, 'frames')
    out_path = os.path.join(workdir, 'out.mp4')
    # ffmpeg_cmd = f'ffmpeg -y -f image2 -framerate 25 -pattern_type glob -i "{in_dir}/*.jpg"  -pix_fmt yuv420p -c:v libx264 -x264opts keyint=25:min-keyint=25:scenecut=-1 -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" {out_path}'
    ffmpeg_cmd = f'ffmpeg -y -f image2 -framerate 25 -pattern_type glob -i "{in_dir}/*.jpg"  -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" {out_path}'
    os.system(ffmpeg_cmd)