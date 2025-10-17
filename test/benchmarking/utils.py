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
import re
import shutil
import subprocess
import time
from multiprocessing import cpu_count

import psutil


def create_logger(name, path):
    import logging

    os.makedirs(os.path.dirname(path), exist_ok=True)

    logger = logging.getLogger(name)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    log_handler = logging.FileHandler(path, "w")
    log_handler.setLevel(logging.INFO)
    log_handler.setFormatter(
        logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")
    )
    logger.addHandler(log_handler)
    logger.setLevel(logging.DEBUG)

    return logger


def clean_directory(directory):
    # purge entire dir
    if os.path.isdir(directory):
        try:
            shutil.rmtree(directory)
        except Exception as e:
            print(f"Failed to delete {directory}. Reason: {e}")

    # re-create a path
    os.makedirs(directory, exist_ok=True)


def get_file_desc(fn):
    pattern = re.compile(r"_(?P<width>\d+)x(?P<height>\d+)(?:_(?P<fps>\d+))?")
    match = pattern.search(fn)
    if match:
        width = int(match.group("width"))
        height = int(match.group("height"))
        fps = int(match.group("fps")) if match.group("fps") else 30
        return width, height, fps
    return None, None, None


def get_max_workers(max_workers: int) -> int:
    if max_workers < 1:
        # leave some CPU power to the system
        max_workers = max(1, int(cpu_count() * 0.9), cpu_count() - 4)
    return min(max_workers, int(cpu_count() * 2))


def get_cmd_times(cmd, passes=1, poll_interval=0.01):
    """
    Extract encoder and speed from a single path part.

    Args:
        cmd: command line to run, as single string for shell=True usage
        passes: number of iterations
        poll_interval: psutil polling interval in seconds

    Returns:
        Tuple of (user_time, sys_time, wallclock_time), all in seconds
    """

    user_time = 0.0
    sys_time = 0.0
    wallclock_time = 0.0

    time_s = time.time()
    for _ in range(passes):
        p = psutil.Popen(
            cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

        cpu_user = 0.0
        cpu_sys = 0.0
        while p.poll() is None:
            try:
                times = p.cpu_times()
                cpu_user = times.user
                cpu_sys = times.system
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                # The process may have terminated in between calls
                pass
            time.sleep(poll_interval)

        user_time += cpu_user
        sys_time += cpu_sys

    wallclock_time = time.time() - time_s

    return user_time / passes, sys_time / passes, wallclock_time / passes
