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
import signal
import subprocess
import sys
from concurrent.futures import as_completed, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

import pandas as pd

import utils
from config_manager import ConfigManager

from format_conversion import convert_to_png, convert_to_y4m, convert_to_yuv
from tqdm import tqdm

config_path = sys.argv[1] if len(sys.argv) > 1 else None
config_manager = ConfigManager(config_path=config_path)
PATHS = config_manager.get_paths()
BINARIES = config_manager.get_binaries()
ENCODER_SETTINGS = config_manager.get_encoder_settings()
COMMON_SETTINGS = config_manager.get_common_settings()
SETTINGS = config_manager.get_settings()

DRY_RUN_MODE = SETTINGS.get("dry_run", False)
MAX_PROC = SETTINGS.get("max_processes", 1)

SOURCE_DATA_DIR: str = PATHS["source_data_dir"]
INPUT_DIRS: Dict[str, str] = {
    "y4m": PATHS["source_y4m_dir"],
    "yuv": PATHS["source_yuv_dir"],
    "png": PATHS["source_png_dir"],
}

ENCODED_DIR: str = PATHS["encoded_dir"]
ENC_LOG_PATH: str = PATHS["enc_log_path"]
ENC_CSV_PATH: str = PATHS["enc_csv_path"]

enc_logger = utils.create_logger("enc_logger", ENC_LOG_PATH)


@dataclass
class EncodeTask:
    encoder_type: str
    encoder_name: str
    speed: int
    quality: int
    nthreads: int
    input_file: str


@dataclass
class EncodeResult:
    output_file: str
    runtime: float
    input_size: int
    output_size: int
    compression_ratio: float


def encode_file(task: EncodeTask, output_dir: str) -> EncodeResult:
    """Encode a single file with the specified parameters"""

    codec_settings = COMMON_SETTINGS[task.encoder_type]

    # re-implement monochrome if needed
    assert not codec_settings[task.encoder_name].get("monochrome", False)

    dir = INPUT_DIRS[codec_settings[task.encoder_name]["input_extension"]]
    input_path: str = os.path.join(dir, task.input_file)
    # passes enables more accurate runtime measurements
    passes = codec_settings[task.encoder_name].get("passes", 1)

    filename_without_extension: str = os.path.splitext(task.input_file)[0]
    extension: str = codec_settings["encode_extension"]

    width, height, fps = utils.get_file_desc(filename_without_extension)

    if width and height and fps:
        kbps = int(width * height * fps * 1.5 * 8 / (task.quality * 1000))
    else:
        kbps = 0

    encoded_file: str = f"{filename_without_extension}.{extension}"
    encoded_path: str = os.path.join(output_dir, encoded_file)

    enc_settings = ENCODER_SETTINGS[task.encoder_type][task.encoder_name]
    encoder_bin_name: str = enc_settings["encoder"]
    cfg_path: str = enc_settings.get("cfg_path", "")

    command: str = enc_settings["command"].format(
        binary_dir=BINARIES[encoder_bin_name],
        q=task.quality,
        kbps=kbps,
        speed=task.speed,
        nthreads=task.nthreads,
        input_path=input_path,
        width=width,
        height=height,
        fps=fps,
        cfg_path=cfg_path,
        output_path=encoded_path,
    )

    # Get input file size
    input_size = os.path.getsize(input_path)
    enc_logger.info(command)

    if DRY_RUN_MODE:
        return EncodeResult(
            output_file=encoded_path,
            runtime=0.0,
            output_size=0,
            input_size=input_size,
            compression_ratio=0.0,
        )

    try:
        runtime, _, _ = utils.get_cmd_times(command, passes)

        # Get output file size and calculate compression ratio
        output_size = os.path.getsize(encoded_path)
        compression_ratio = input_size / output_size if output_size > 0 else 0.0

        return EncodeResult(
            output_file=encoded_path,
            runtime=runtime,
            output_size=output_size,
            input_size=input_size,
            compression_ratio=compression_ratio,
        )
    except subprocess.CalledProcessError as e:
        if "Signals.SIGINT" in e:
            raise KeyboardInterrupt
        else:
            enc_logger.exception(f"Error: {e.stderr.decode()}")
            raise RuntimeError("Encoding failed") from e
    except KeyError as e:
        enc_logger.exception(f"Configuration error: {e}")
        raise ValueError("Invalid configuration") from e


def get_input_files(encoder_type: str, encoder_name: str) -> List[str]:
    codec_settings = COMMON_SETTINGS[encoder_type]

    # re-implement monochrome if needed
    assert not codec_settings[encoder_name].get("monochrome", False)

    ext: str = codec_settings[encoder_name]["input_extension"]
    return [x for x in os.listdir(INPUT_DIRS[ext]) if x.endswith(ext)]


def get_quality_values(encoder_type: str, encoder_name: str) -> List[int]:
    dir_format: Dict[str, Any] = COMMON_SETTINGS[encoder_type].get("dir_format", {})
    quality_param: str = dir_format.get("quality_param", "q")
    quality_values_key: str = f"{quality_param}_values"
    quality_values: List[int] = COMMON_SETTINGS[encoder_type][encoder_name].get(
        quality_values_key
    )
    return quality_values


def get_output_dir(task: EncodeTask) -> str:
    base_dir: str = ENCODED_DIR

    dir_format = COMMON_SETTINGS[task.encoder_type].get("dir_format", {})

    speed_suffix = dir_format.get("speed_suffix", "")
    if speed_suffix:
        name_suffix = speed_suffix.format(speed=task.speed)
    else:
        name_suffix = ""

    quality_param = dir_format.get("quality_param", "q")
    quality_str = f"{quality_param}{task.quality}"

    base_output_dir: str = os.path.join(base_dir, f"{task.encoder_name}{name_suffix}")
    output_dir: str = os.path.join(base_output_dir, f"{quality_str}_t{task.nthreads}")
    return output_dir


def execute_encode_job(task: EncodeTask) -> Tuple[EncodeTask, EncodeResult]:
    """Execute a single encode job and return the result"""
    output_dir = get_output_dir(task)

    if not DRY_RUN_MODE:
        os.makedirs(output_dir, exist_ok=True)

    return task, encode_file(task, output_dir)


def create_encode_jobs() -> List[EncodeTask]:
    """Create all encode jobs for all combinations of parameters"""
    jobs = []

    for encoder_type, encoder_settings in ENCODER_SETTINGS.items():
        for encoder_name, encoder_config in encoder_settings.items():
            if encoder_name not in config_manager.get_codecs().get(
                "allowed_codecs", []
            ):
                continue

            settings = COMMON_SETTINGS[encoder_type][encoder_name]
            speed_values = settings["speed_values"]
            quality_values = get_quality_values(encoder_type, encoder_name)
            nthreads = encoder_config["nthreads"]
            input_files = get_input_files(encoder_type, encoder_name)

            for speed in speed_values:
                for quality in quality_values:
                    for nthread in nthreads:
                        for input_file in input_files:
                            jobs.append(
                                EncodeTask(
                                    encoder_type=encoder_type,
                                    encoder_name=encoder_name,
                                    speed=speed,
                                    quality=quality,
                                    input_file=input_file,
                                    nthreads=nthread,
                                )
                            )

    return jobs


def detect_source_format(source_data_dir: str) -> str:
    """Detect the file format of the source data directory"""
    file_formats = ["y4m", "yuv", "png"]
    extensions_on_disk = [f.split(".")[-1] for f in os.listdir(source_data_dir)]
    for file_format in file_formats:
        if file_format in extensions_on_disk:
            return file_format
    raise ValueError(f"Could not detect file format of {source_data_dir}")


def generate_input_data_reformats(input_ext, source_data_dir, target_data_dirs):
    filenames = [
        os.path.join(source_data_dir, f)
        for f in os.listdir(source_data_dir)
        if f.endswith(input_ext)
    ]

    formats = set()
    for _, codec_settings in COMMON_SETTINGS.items():
        for codec_name in config_manager.get_codecs()["allowed_codecs"]:
            if codec_name in codec_settings:
                formats.add(codec_settings[codec_name]["input_extension"])

    allow_metrics = {"vmaf": False, "ssimulacra2": False, "mssim": False}
    for m in config_manager.get_metrics().get("allowed_metrics", []):
        allow_metrics[m] = True

    if allow_metrics["ssimulacra2"] or allow_metrics["mssim"]:
        formats.add("png")
    if allow_metrics["vmaf"] and not ("yuv" in formats):
        formats.add("y4m")

    _ = [os.makedirs(x, exist_ok=True) for x in target_data_dirs.values()]
    converters = {
        "y4m": convert_to_y4m,
        "yuv": convert_to_yuv,
        "png": convert_to_png,
    }
    for filename in filenames:
        for fmt in formats:
            target_data_dir = target_data_dirs[fmt]
            converters[fmt](filename, target_data_dir, enc_logger)


def main() -> None:
    """Main function to execute all encode jobs and log results"""

    os.makedirs(os.path.dirname(ENC_CSV_PATH), exist_ok=True)

    input_format = detect_source_format(SOURCE_DATA_DIR)
    generate_input_data_reformats(input_format, SOURCE_DATA_DIR, INPUT_DIRS)

    if DRY_RUN_MODE:
        enc_logger.info("#" + "=" * 59)
        enc_logger.info("# DRY-RUN MODE: Commands will be logged but not executed")
        enc_logger.info("#" + "=" * 59)
    else:
        utils.clean_directory(ENCODED_DIR)

    # Create all encode jobs
    jobs = create_encode_jobs()
    enc_logger.info(f"Created {len(jobs)} encode jobs")

    if not jobs:
        enc_logger.info("No encode jobs to run")
        return

    print(f"Output CSV: {ENC_CSV_PATH}")
    need_csv_header = True

    # be nice when using multiprocessing
    os.nice(10)
    max_workers = utils.get_max_workers(MAX_PROC)

    # Execute jobs in threadpool with progress bar
    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        def signal_handler(sig, frame):
            print("Received Ctrl-C, shutting down...")
            executor.shutdown(wait=False, cancel_futures=True)
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        futures = [executor.submit(execute_encode_job, job) for job in jobs]

        for future in tqdm(
            as_completed(futures), total=len(jobs), desc="Encoding files"
        ):
            res = "ok"
            try:
                job, result = future.result()
                pd.DataFrame([{**asdict(job), **asdict(result)}]).to_csv(
                    ENC_CSV_PATH,
                    index=False,
                    header=need_csv_header,
                    mode="w" if need_csv_header else "a",
                )
                need_csv_header = False
            except Exception as e:
                res = str(e)

            log_msg = (
                f"Completed: {job.encoder_name} speed={job.speed} "
                f"quality={job.quality} threads={job.nthreads} "
                f"file={job.input_file} "
                f"input_size={result.input_size} bytes "
                f"output_size={result.output_size} bytes "
                f"compression_ratio={result.compression_ratio:.3f} -> {res}"
            )
            enc_logger.info(log_msg)

    # Log summary statistics
    enc_logger.info("")
    enc_logger.info("Encoding Summary:")
    enc_logger.info(f"Total jobs: {len(jobs)}")
    enc_logger.info(f"Results saved to: {ENC_CSV_PATH}")


if __name__ == "__main__":
    main()
