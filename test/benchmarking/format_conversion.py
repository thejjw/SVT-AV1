# Copyright(c) 2025 Meta Platforms, Inc. and affiliates.
#
# This source code is subject to the terms of the BSD 2 Clause License and
# the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
# was not distributed with this source code in the LICENSE file, you can
# obtain it at https://www.aomedia.org/license/software-license. If the
# Alliance for Open Media Patent License 1.0 was not distributed with this
# source code in the PATENTS file, you can obtain it at
# https://www.aomedia.org/license/patent-license.

import os
import subprocess
import sys
from logging import Logger
from pathlib import Path
from typing import Any, Dict

from config_manager import ConfigManager

config_path = sys.argv[1] if len(sys.argv) > 1 else None
config_manager = ConfigManager(config_path=config_path)

BINARIES: Dict[str, str] = config_manager.get_binaries()
COMMON_SETTINGS: Dict[str, Dict[str, Any]] = config_manager.get_common_settings()

FFMPEG_BIN: str = BINARIES["ffmpeg"]
FFPROBE_BIN: str = BINARIES["ffprobe"]


# convert to y4m limited color range, should be VMAF compliant
def convert_to_y4m(file: str, target_dir: str, logger: Logger):
    output_file = os.path.join(target_dir, Path(file).stem + ".y4m")

    if os.path.exists(output_file):
        return output_file

    if file.endswith(".y4m"):
        try:
            os.symlink(file, output_file)
            logger.info(f"Linked Y4M file: {file} -> {output_file}")
        except Exception as e:
            logger.error(f"Error linking Y4M file {file} - {e}")
    else:
        subprocess.run(
            [
                FFMPEG_BIN,
                "-loglevel",
                "error",
                "-i",
                file,
                "-vf",
                "scale=in_color_matrix=bt601:in_range=pc:out_color_matrix=bt601:out_range=pc:flags=lanczos+accurate_rnd+full_chroma_int:sws_dither=none:param0=5,format=yuv420p",
                "-pix_fmt",
                "yuv420p",
                "-color_range",
                "pc",
                "-y",
                output_file,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    return output_file


def convert_to_png(file: str, target_dir: str, logger: Logger):
    output_file = os.path.join(target_dir, Path(file).stem + ".png")

    if os.path.exists(output_file):
        return output_file

    if file.endswith(".png"):
        try:
            os.symlink(file, output_file)
            logger.info(f"Linked PNG file: {file} -> {output_file}")
        except Exception as e:
            logger.error(f"Error linking PNG file {file} - {e}")
    else:
        subprocess.run(
            [
                FFMPEG_BIN,
                "-loglevel",
                "error",
                "-i",
                file,
                "-vf",
                "scale=in_color_matrix=bt601:in_range=pc:out_color_matrix=bt601:out_range=pc:flags=lanczos+accurate_rnd+full_chroma_int:sws_dither=none:param0=5,format=rgb24",
                "-f",
                "image2",
                "-update",
                "1",
                "-frames:v",
                "1",
                "-y",
                output_file,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    return output_file


def get_resolution(filename):
    """
    Given a file in the png or y4m format, will return a tuple for (width, height)
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0:s=x",
        filename,
    ]
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    res = result.stdout.strip()
    if "x" in res:
        width, height = map(int, res.split("x"))
        return width, height
    else:
        raise ValueError(f"Could not get resolution for {filename}")


def convert_to_yuv(file: str, target_dir: str, logger: Logger):
    output_file = os.path.join(target_dir, Path(file).stem)

    if file.endswith(".yuv"):
        # If input is already YUV, just copy it
        try:
            target_path = output_file + ".yuv"
            if not os.path.exists(target_path):
                os.symlink(file, target_path)
                logger.info(f"Linked YUV file: {file} -> {target_path}")
        except Exception as e:
            logger.error(f"Error linking YUV file {file} - {e}")
        return target_path

    if file.endswith(".y4m") or file.endswith(".png"):
        width, height = get_resolution(file)
        target_path = output_file + f"_{width}x{height}.yuv"
        if not os.path.exists(target_path):
            # Convert Y4M/PNG to YUV
            subprocess.run(
                [
                    FFMPEG_BIN,
                    "-loglevel",
                    "error",
                    "-i",
                    file,
                    "-vf",
                    "scale=in_color_matrix=bt601:in_range=pc:out_color_matrix=bt601:out_range=pc:flags=lanczos+accurate_rnd+full_chroma_int:sws_dither=none:param0=5,format=yuv420p",
                    "-pix_fmt",
                    "yuv420p",
                    "-color_range",
                    "pc",
                    "-y",
                    target_path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        return target_path

    logger.error(f"Unexpected file type - {file}")
    return None
