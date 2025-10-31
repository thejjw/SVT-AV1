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
from typing import Dict, List, Tuple

import pandas as pd

import utils
from config_manager import ConfigManager

from encode import detect_source_format, generate_input_data_reformats
from format_conversion import convert_to_png, convert_to_y4m
from qm import QualityMetricsCalculator
from tqdm import tqdm

config_path = sys.argv[1] if len(sys.argv) > 1 else None
config_manager = ConfigManager(config_path=config_path)
PATHS = config_manager.get_paths()
BINARIES = config_manager.get_binaries()
COMMON_SETTINGS = config_manager.get_common_settings()
ENCODER_SETTINGS = config_manager.get_encoder_settings()
DECODER_SETTINGS = config_manager.get_decoder_settings()
SETTINGS = config_manager.get_settings()

DRY_RUN_MODE = SETTINGS.get("dry_run", False)
MAX_PROC = SETTINGS.get("max_processes", 1)

ENCODED_DIR: str = PATHS["encoded_dir"]
DECODED_DIR: str = PATHS["decoded_dir"]
QM_DIR: str = PATHS["qm_dir"]
CONVERSION_DIR: str = PATHS["conversion_dir"]
DEC_LOG_PATH: str = PATHS["dec_log_path"]
DEC_CSV_PATH: str = PATHS["dec_csv_path"]
QM_LOG_PATH: str = PATHS["qm_log_path"]

SOURCE_DATA_DIR: str = PATHS["source_data_dir"]
SOURCE_PNG_DIR: str = PATHS["source_png_dir"]
SOURCE_Y4M_DIR: str = PATHS["source_y4m_dir"]
SOURCE_YUV_DIR: str = PATHS["source_yuv_dir"]

dec_logger = utils.create_logger("dec_logger", DEC_LOG_PATH)
qm_logger = utils.create_logger("qm_logger", QM_LOG_PATH)


@dataclass
class DecodeTask:
    codec_type: str
    codec_name: str
    speed: int
    quality: int
    nthreads: int
    input_file: str


@dataclass
class DecodeResult:
    codec_type: str
    codec_name: str
    speed: int
    quality: int
    nthreads: int
    decode_time: int
    input_file: str
    output_file: str
    # Quality metrics
    ssimulacra2: float
    psnr_y: float
    psnr_cb: float
    psnr_cr: float
    ssim: float
    ms_ssim: float
    vmaf: float
    vmaf_neg: float


def decode_file(
    codec_type: str,
    codec_name: str,
    input_file: str,
    nthreads: int,
    output_dir: str,
) -> Tuple[str, str, int, str, float]:
    """Decode a single file with the specified parameters"""

    filename_without_extension: str = os.path.basename(os.path.splitext(input_file)[0])
    extension: str = COMMON_SETTINGS[codec_type]["decode_extension"]

    width, height, fps = utils.get_file_desc(filename_without_extension)

    output_file: str = f"{filename_without_extension}.{extension}"
    output_path = os.path.join(output_dir, output_file)

    dec_settings = DECODER_SETTINGS[codec_type][codec_name]
    decoder_bin_name = dec_settings["decoder"]

    command = dec_settings["command"].format(
        binary_dir=BINARIES[decoder_bin_name],
        input_path=input_file,
        width=width,
        height=height,
        fps=fps,
        output_path=output_path,
        nthreads=nthreads,
    )

    dec_logger.info(command)

    if DRY_RUN_MODE:
        return codec_name, input_file, nthreads, output_path, 0.0

    # passes enables more accurate runtime measurements
    passes = COMMON_SETTINGS[codec_type][codec_name].get("passes", 1)
    try:
        runtime, _, _ = utils.get_cmd_times(command, passes)
        return codec_name, input_file, nthreads, output_path, runtime
    except subprocess.CalledProcessError as e:
        if "Signals.SIGINT" in e:
            raise KeyboardInterrupt
        else:
            dec_logger.exception(f"Error: {e.stderr.decode()}")
            raise RuntimeError("Decoding failed") from e
    except KeyError as e:
        dec_logger.exception(f"Configuration error: {e}")
        raise ValueError("Invalid configuration") from e


def get_source_file_path(filename: str) -> str:
    """Get the source file path for quality metrics comparison"""
    if SOURCE_PNG_DIR:
        # Extract base name without extension
        base_name = os.path.splitext(filename)[0]
        # Determine file extension for quality metrics
        for file in os.listdir(SOURCE_PNG_DIR):
            if file.lower().endswith(".pgm") or file.lower().endswith(".png"):
                # Look for matching source file
                source_filename = base_name + os.path.splitext(file)[1]
                source_path = os.path.join(SOURCE_PNG_DIR, source_filename)
                if os.path.exists(source_path):
                    return source_path
    return ""


def get_source_y4m_path(filename: str) -> str:
    """Get the source Y4M file path for VMAF metrics"""
    if SOURCE_Y4M_DIR:
        base_name = os.path.splitext(filename)[0]
        y4m_filename = base_name + ".y4m"
        source_y4m_path = os.path.join(SOURCE_Y4M_DIR, y4m_filename)
        if os.path.exists(source_y4m_path):
            return source_y4m_path
    return ""


def get_source_yuv_path(filename: str) -> str:
    """Get the source YUV file path for PSNR metrics"""
    if SOURCE_YUV_DIR:
        base_name = os.path.splitext(filename)[0]
        yuv_filename = base_name + ".yuv"
        source_yuv_path = os.path.join(SOURCE_YUV_DIR, yuv_filename)
        if os.path.exists(source_yuv_path):
            return source_yuv_path
    return ""


def calculate_single_file_quality_metrics(
    decoded_file: str,
    sub_dir_name: str,
) -> Dict[str, float]:
    """Calculate quality metrics for a single decoded file"""

    filename = os.path.basename(decoded_file)

    # Get source files
    ref_file = get_source_file_path(filename)
    ref_y4m_file = get_source_y4m_path(filename)
    ref_yuv_file = get_source_yuv_path(filename)

    if not ref_file and not ref_y4m_file and not ref_yuv_file:
        dec_logger.warning(f"No source files found for {filename}")
        return {}

    # Prepare converted files for metrics
    conv_dir = os.path.join(CONVERSION_DIR, sub_dir_name)
    os.makedirs(conv_dir, exist_ok=True)

    # Create quality metrics calculator
    allow_metrics = {"vmaf": False, "ssimulacra2": False, "mssim": False}
    for m in config_manager.get_metrics().get("allowed_metrics", []):
        allow_metrics[m] = True

    need_png = False
    need_y4m = False

    if allow_metrics["ssimulacra2"] or allow_metrics["mssim"]:
        need_png = True
    if allow_metrics["vmaf"]:
        need_y4m = not (ref_yuv_file and filename.endswith(".yuv"))

    dist_png_file = (
        convert_to_png(decoded_file, conv_dir, dec_logger) if need_png else ""
    )
    dist_y4m_file = (
        convert_to_y4m(decoded_file, conv_dir, dec_logger) if need_y4m else ""
    )
    dist_yuv_file = decoded_file if decoded_file.endswith(".yuv") else ""

    calculator = QualityMetricsCalculator(
        SOURCE_PNG_DIR,
        SOURCE_Y4M_DIR,
        SOURCE_YUV_DIR,
        os.path.dirname(dist_png_file),
        os.path.dirname(dist_y4m_file),
        os.path.dirname(dist_yuv_file),
        qm_logger,
        BINARIES,
        allow_metrics,
        config_manager.get_metrics().get("aom_ctc_model", "v6.0"),
        os.path.join(QM_DIR, sub_dir_name),
    )

    # Calculate metrics for this single file
    metrics = calculator.calculate_single_file_metrics(
        ref_file,
        dist_png_file if os.path.exists(dist_png_file) else None,
        ref_y4m_file,
        dist_y4m_file if os.path.exists(dist_y4m_file) else None,
        ref_yuv_file,
        dist_yuv_file if os.path.exists(dist_yuv_file) else None,
    )

    return metrics


def get_output_dir(
    codec_type: str,
    codec_name: str,
    speed: int,
    quality: int,
    nthreads: int,
) -> str:
    """Get output directory path following the same pattern as encoding"""
    base_dir = DECODED_DIR

    dir_format = COMMON_SETTINGS[codec_type].get("dir_format", {})

    speed_suffix = dir_format.get("speed_suffix", "")
    if speed_suffix:
        name_suffix = speed_suffix.format(speed=speed)
    else:
        name_suffix = ""

    quality_param = dir_format.get("quality_param", "q")
    quality_str = f"{quality_param}{quality}"

    base_output_dir = os.path.join(base_dir, f"{codec_name}{name_suffix}")
    output_dir = os.path.join(base_output_dir, f"{quality_str}_t{nthreads}")
    return output_dir


def get_sub_dir_name(
    codec_type: str,
    codec_name: str,
    speed: int,
    quality: int,
    nthreads: int,
) -> str:
    """Get subdirectory name for organizing conversions"""
    dir_format = COMMON_SETTINGS[codec_type].get("dir_format", {})

    speed_suffix = f"_speed{speed}" if dir_format.get("speed_suffix") else ""
    quality_param = dir_format.get("quality_param", "q")

    return f"{codec_name}{speed_suffix}/{quality_param}{quality}_t{nthreads}"


def execute_decode_job(task: DecodeTask) -> Tuple[DecodeTask, DecodeResult]:
    """Execute a single decode + quality metrics job and return the result"""

    output_dir = get_output_dir(
        task.codec_type,
        task.codec_name,
        task.speed,
        task.quality,
        task.nthreads,
    )

    if not DRY_RUN_MODE:
        os.makedirs(output_dir, exist_ok=True)

    # Decode the file
    codec_name, input_file, nthreads, decoded_path, decode_time = decode_file(
        task.codec_type,
        task.codec_name,
        task.input_file,
        task.nthreads,
        output_dir,
    )

    # Calculate quality metrics for the decoded file
    sub_dir_name = get_sub_dir_name(
        task.codec_type,
        task.codec_name,
        task.speed,
        task.quality,
        task.nthreads,
    )

    quality_metrics = {}
    if not DRY_RUN_MODE:
        quality_metrics = calculate_single_file_quality_metrics(
            decoded_path, sub_dir_name
        )

    # Build result dictionary with all quality metrics as separate columns
    result = DecodeResult(
        codec_type=task.codec_type,
        codec_name=codec_name,
        speed=task.speed,
        quality=task.quality,
        nthreads=nthreads,
        decode_time=decode_time,
        input_file=os.path.basename(input_file),
        output_file=os.path.basename(decoded_path),
        # Quality metrics
        ssimulacra2=quality_metrics.get("ssimulacra2", None),
        psnr_y=quality_metrics.get("psnr_y", None),
        psnr_cb=quality_metrics.get("psnr_cb", None),
        psnr_cr=quality_metrics.get("psnr_cr", None),
        ssim=quality_metrics.get("float_ssim", None),
        ms_ssim=quality_metrics.get("float_ms_ssim", None),
        vmaf=quality_metrics.get("vmaf", None),
        vmaf_neg=quality_metrics.get("vmaf_neg", None),
    )

    return task, result


def create_decode_jobs() -> List[DecodeTask]:
    """Create all decode jobs for all combinations of parameters"""
    jobs = []

    for codec_type, codecs in DECODER_SETTINGS.items():
        for codec_name, _ in codecs.items():
            if codec_name not in config_manager.get_codecs().get("allowed_codecs", []):
                continue

            dir_format = COMMON_SETTINGS[codec_type].get("dir_format", {})
            quality_param = dir_format.get("quality_param", "q")
            quality_values_key = f"{quality_param}_values"

            quality_values = COMMON_SETTINGS[codec_type][codec_name].get(
                quality_values_key, []
            )
            speed_values = COMMON_SETTINGS[codec_type][codec_name]["speed_values"]
            nthreads_list = ENCODER_SETTINGS[codec_type][codec_name]["nthreads"]

            for speed in speed_values:
                for quality in quality_values:
                    for nthreads in nthreads_list:
                        # Determine directory structure
                        speed_suffix = (
                            f"_speed{speed}" if dir_format.get("speed_suffix") else ""
                        )
                        sub_dir_name = f"{codec_name}{speed_suffix}/{quality_param}{quality}_t{nthreads}"

                        input_dir = os.path.join(ENCODED_DIR, sub_dir_name)

                        if not os.path.exists(input_dir):
                            continue

                        # Create jobs for each file in the input directory
                        for filename in sorted(os.listdir(input_dir)):
                            input_file = os.path.join(input_dir, filename)
                            if os.path.isfile(input_file):
                                jobs.append(
                                    DecodeTask(
                                        codec_type=codec_type,
                                        codec_name=codec_name,
                                        speed=speed,
                                        quality=quality,
                                        nthreads=nthreads,
                                        input_file=input_file,
                                    )
                                )

    return jobs


def main() -> None:
    """Main function to execute all decode jobs and log results"""

    os.makedirs(os.path.dirname(DEC_CSV_PATH), exist_ok=True)

    input_format = detect_source_format(SOURCE_DATA_DIR)
    generate_input_data_reformats(
        input_format,
        SOURCE_DATA_DIR,
        {
            "y4m": SOURCE_Y4M_DIR,
            "yuv": SOURCE_YUV_DIR,
            "png": SOURCE_PNG_DIR,
        },
    )

    if DRY_RUN_MODE:
        dec_logger.info("#" + "=" * 59)
        dec_logger.info("# DRY-RUN MODE: Commands will be logged but not executed")
        dec_logger.info("#" + "=" * 59)
    else:
        utils.clean_directory(DECODED_DIR)
        utils.clean_directory(QM_DIR)
        utils.clean_directory(CONVERSION_DIR)

    # Create all decode jobs
    jobs = create_decode_jobs()
    dec_logger.info(f"Created {len(jobs)} decode jobs")

    if not jobs:
        dec_logger.info("No decode jobs to run")
        return

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

        futures = [executor.submit(execute_decode_job, job) for job in jobs]

        for future in tqdm(
            as_completed(futures), total=len(jobs), desc="Decoding files"
        ):
            res = "ok"
            try:
                job, result = future.result()
                pd.DataFrame([asdict(result)]).to_csv(
                    DEC_CSV_PATH,
                    index=False,
                    header=need_csv_header,
                    mode="w" if need_csv_header else "a",
                )
                need_csv_header = False
            except Exception as e:
                res = str(e)

            log_msg = (
                f"Completed: {job.codec_name} speed={job.speed} "
                f"quality={job.quality} threads={job.nthreads} "
                f"file={os.path.basename(job.input_file)} -> {res}"
            )
            dec_logger.info(log_msg)

    # Log summary statistics
    dec_logger.info("")
    dec_logger.info("Decoding Summary:")
    dec_logger.info(f"Total jobs: {len(jobs)}")
    dec_logger.info(f"Results saved to: {DEC_CSV_PATH}")


if __name__ == "__main__":
    main()
