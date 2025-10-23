# Copyright(c) 2025 Meta Platforms, Inc. and affiliates.
#
# This source code is subject to the terms of the BSD 2 Clause License and
# the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
# was not distributed with this source code in the LICENSE file, you can
# obtain it at https://www.aomedia.org/license/software-license. If the
# Alliance for Open Media Patent License 1.0 was not distributed with this
# source code in the PATENTS file, you can obtain it at
# https://www.aomedia.org/license/patent-license.

"""
Run using one of the following options:

  # Use default --read_qm mode; will read the xml and ssimulacra files for quality metrics
  python summary.py config.yaml

  # Read the csv files provided for encoding, decoding, and quality metrics
  python summary.py config.yaml --use_enc_qm_logs

  # Process per-image logs where each per image contains all necessary columns;
  # Used for merging data from different datasets
  python summary.py config.yaml --use_per_image_log file1.csv file2.csv
"""

import argparse
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd

from analysis_and_plotting import create_per_image_rd_plots, run_bd_rate_analysis
from config_manager import ConfigManager
from data_readers import DataReaders

QUALITY_METRICS = [
    "SSIMULACRA2",
    "psnr",
    "psnr_y",
    "psnr_cb",
    "psnr_cr",
    "ssim",
    "ms_ssim",
    "vmaf",
    "vmaf_neg",
]

# Configurations will be loaded after parsing arguments
config_manager = None
PATHS = None
COMMON_SETTINGS = None
SETTINGS = None
ENCODED_DIR = None
ENCODING_LOG_FILE = None
ENCODING_CSV_FILE = None
DECODING_LOG_FILE = None
DECODING_CSV_FILE = None
QM_LOG_FILE = None
OUTPUT_DIR = None
PER_IMAGE_FILE = None
AV1_ENCODER_NAMES = None

# Constants
BYTES_TO_MB = 1024 * 1024  # Conversion factor from bytes to megabytes


def get_total_file_size(directory: str) -> float:
    total_size = 0
    for root, _, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(root, file)
            try:
                total_size += os.path.getsize(filepath)
            except OSError as e:
                print(f"Error accessing file {filepath}: {e}")
    return total_size


def get_encoder_order() -> List[str]:
    """Get the ordered list of encoder names with speed suffixes"""

    encoder_order = []
    for _, settings in COMMON_SETTINGS.items():
        for codec_name in settings:
            if (
                isinstance(settings[codec_name], dict)
                and "speed_values" in settings[codec_name]
            ):
                for speed in settings[codec_name]["speed_values"]:
                    encoder_order.append(f"{codec_name}_speed{speed}")
    return encoder_order


def sort_encoding_results(
    encoding_results: Dict[Tuple[str, int, int, int], float],
    encoder_order: List[str],
    qm_results: Optional[Dict[Tuple[str, int, int, int], Dict[str, float]]] = None,
) -> List[Tuple[Tuple[str, int, int, int], float]]:
    """Sort encoding results by encoder order and quality metrics"""

    sorted_encoding_results = []
    for encoder in encoder_order:
        # Collect all matching results for this encoder
        encoder_results = []

        for key, value in encoding_results.items():
            if (
                key[0] == encoder.replace("_speed", "")
                or f"{key[0]}_speed{key[1]}" == encoder
            ):
                encoder_results.append((key, value))

        # Sort the results for this encoder
        if encoder_results:
            # Sort by PSNR_Y in descending order (higher quality first)
            def get_psnr_y_for_sorting(result_tuple):
                key = result_tuple[0]  # (encoder, speed, quality, threads)

                # Get PSNR_Y value from quality metrics
                if qm_results and key in qm_results:
                    qm_values = qm_results[key]
                    # Try to get psnr_y (for libvmaf) or psnr (for traditional metrics)
                    psnr_y = qm_values.get("psnr_y", qm_values.get("psnr", 0))
                    return (key[1], -psnr_y, key[3])  # speed, -psnr_y (desc), threads
                else:
                    # Fallback to quality parameter if no QM data available
                    return (key[1], -key[2], key[3])  # speed, -quality (desc), threads

            encoder_results.sort(key=get_psnr_y_for_sorting)

            # Add sorted results to the final list
            sorted_encoding_results.extend(encoder_results)

    return sorted_encoding_results


def get_csv_headers() -> List[str]:
    """Return CSV headers for combined QM format."""
    return [
        "Encoder",
        "Speed",
        "QP",
        "Threads",
        "File Size (MB)",
        "SSIMULACRA2",
        "psnr_y",
        "psnr_cb",
        "psnr_cr",
        "ssim",
        "ms_ssim",
        "vmaf",
        "vmaf_neg",
        "Encoding Time (s)",
        "Decoding Time (s)",
    ]


def setup_argument_parser():
    """Set up and parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Image Coding Summary Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python summary.py config.yaml                          # Use default --read_qm mode
  python summary.py config.yaml --use_enc_qm_logs        # Read encoding CSV and QM CSV files
  python summary.py config.yaml --use_per_image_log file1.csv file2.csv  # Process per-image logs
        """,
    )

    # Required config file argument
    parser.add_argument(
        "config_file",
        help="Path to the configuration YAML file that defines paths, encoders, and settings",
    )

    # Mode selection group - make it mutually exclusive but not required (--read_qm is default)
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--use_enc_qm_logs",
        action="store_true",
        help="Read encoding CSV and QM CSV files, merge them and run plotting/BD-rate analysis",
    )
    group.add_argument(
        "--use_per_image_log",
        nargs="+",
        help="Accept list of CSV files with per_image results and combine them",
    )
    group.add_argument(
        "--read_qm",
        action="store_true",
        default=True,
        help="Read quality metrics from qm_dir and create per_image dataframe (default mode)",
    )

    return parser.parse_args()


def process_enc_qm_logs(data_readers: DataReaders) -> pd.DataFrame:
    """
    Process encoding CSV and QM CSV files, merge them and create per_image dataframe.

    Args:
        data_readers: Centralized data reader instance

    Returns:
        DataFrame with per-image results
    """
    print("Using encoding CSV and QM CSV logs mode...")

    # Parse per-image data and save CSV
    encoding_df = data_readers.read_encoding_csv_data()
    decode_qm_df = data_readers.read_decode_qm_csv_data()
    per_image_data = data_readers.merge_enc_dec_qm_dfs(encoding_df, decode_qm_df)

    return per_image_data


def process_per_image_logs(filenames: List[str]) -> pd.DataFrame:
    """
    Process multiple per-image CSV files and combine them.

    Args:
        filenames: List of CSV file paths to process

    Returns:
        DataFrame with combined per-image results
    """
    print(f"Using per-image log files mode with {len(filenames)} files...")

    filenames = [fname.strip() for fname in filenames]
    dfs = [
        DataReaders.read_preprocessed_per_image_data(fname, BYTES_TO_MB)
        for fname in filenames
    ]
    per_image_data = pd.concat(dfs, axis=0)

    if "psnr" in per_image_data.columns and "psnr_y" in per_image_data.columns:
        per_image_data = per_image_data.drop(columns=["psnr_y"])

    # if there are rows with no dataset specified that will be a problem, remove dataset from columns.
    if (
        "dataset" in per_image_data.columns
        and per_image_data["dataset"].isna().sum() > 0
    ):
        per_image_data = per_image_data.drop(columns=["dataset"])

    return per_image_data


def process_qm_directory(data_readers: DataReaders) -> pd.DataFrame:
    """
    Read quality metrics from qm_dir, if available, join them with encoding csv data, to create per_image dataframe.

    Args:
        data_readers: Centralized data reader instance

    Returns:
        DataFrame with per-image results including quality metrics, file sizes, and runtime
    """
    print("Reading quality metrics from qm_dir...")

    # Get qm_dir from config
    qm_dir = PATHS.get("qm_dir")

    # Read quality metrics from qm_dir using centralized reader
    qm_df = data_readers.read_qm_logs_from_dir(qm_dir)

    if qm_df.empty:
        print("No quality metrics found in qm_dir")
        return pd.DataFrame()

    # Try to get encoding CSV data using centralized reader
    encoding_data = data_readers.read_encoding_csv_data()

    # Merge with encoding data if available
    if encoding_data is not None and not encoding_data.empty:
        print("Found encoding CSV data, merging with QM data...")

        # Prepare merge keys for both dataframes
        qm_df_for_merge = qm_df.copy()
        encoding_df_for_merge = encoding_data.copy()

        # Column renaming is now handled in _safe_read_csv, no need for additional renaming
        # encoding_df_for_merge already has standardized column names

        # Merge on image, encoder, speed, quality, threads
        merge_keys = ["image", "encoder", "speed", "quality", "threads"]

        # Check if all merge keys exist in both dataframes
        missing_keys_qm = [
            key for key in merge_keys if key not in qm_df_for_merge.columns
        ]
        missing_keys_enc = [
            key for key in merge_keys if key not in encoding_df_for_merge.columns
        ]
        if "file_size_mb" not in list(encoding_data.columns) + list(
            qm_df_for_merge.columns
        ):
            print(
                "Warning: Encoding CSV missing file_size_mb column, will try to get it from disk"
            )
            qm_df_for_merge = data_readers.add_file_size_from_disk(qm_df_for_merge)

        if missing_keys_qm or missing_keys_enc:
            print(
                f"Warning: Cannot merge - missing keys. QM missing: {missing_keys_qm}, Encoding missing: {missing_keys_enc}"
            )
        else:
            # Perform the merge
            merge_columns = merge_keys + [
                i
                for i in ["encoding_time", "file_size_mb"]
                if i in encoding_df_for_merge.columns
            ]
            per_image_data = pd.merge(
                qm_df_for_merge,
                encoding_df_for_merge[merge_columns],
                on=merge_keys,
                how="left",
            )

            print(
                f"Successfully merged {len(per_image_data)} records with encoding data"
            )

            # For any rows that didn't get file_size_mb from the merge, try to get it from disk
            missing_filesize_mask = per_image_data["file_size_mb"].isna()
            if missing_filesize_mask.sum() > 0:
                print(
                    f"Adding file size from disk for {missing_filesize_mask.sum()} records with missing file size"
                )
                missing_data = per_image_data[missing_filesize_mask].copy()
                missing_with_filesize = data_readers.add_file_size_from_disk(
                    missing_data[qm_df.columns]
                )
                # Update the missing file sizes
                per_image_data.loc[missing_filesize_mask, "file_size_mb"] = (
                    missing_with_filesize["file_size_mb"].values
                )
    else:
        # No encoding CSV available, fall back to file size calculation from disk
        print("No encoding CSV available, calculating file sizes from disk...")
        qm_df = data_readers.add_file_size_from_disk(qm_df)
        per_image_data = qm_df

    return per_image_data


def run_analysis(per_image_data: pd.DataFrame, per_image_csv_path: str) -> None:
    """
    Run BD-rate analysis and create RD plots using per-image data.

    Args:
        per_image_data: DataFrame with per-image results
        per_image_csv_path: Path where the per-image CSV should be saved
    """
    if per_image_data.empty:
        print("No per-image data available for analysis")
        return

    # Save per-image CSV
    per_image_data.to_csv(per_image_csv_path, index=False)
    print(f"Per-image results saved to: {per_image_csv_path}")

    # Create per-image RD plots using new analysis module
    create_per_image_rd_plots(per_image_data, QUALITY_METRICS, OUTPUT_DIR)

    # Run BD-rate analysis using new analysis module if enabled
    run_bd_rate_analysis(
        per_image_csv_path, QUALITY_METRICS, OUTPUT_DIR, config_manager.get_metrics()
    )


def initialize_configuration(config_file: str) -> None:
    """Initialize global configuration variables from config file."""
    global config_manager, PATHS, COMMON_SETTINGS, SETTINGS
    global ENCODED_DIR, ENCODING_LOG_FILE, ENCODING_CSV_FILE, DECODING_LOG_FILE
    global DECODING_CSV_FILE, QM_LOG_FILE, OUTPUT_DIR
    global PER_IMAGE_FILE, AV1_ENCODER_NAMES

    config_manager = ConfigManager(config_path=config_file)
    PATHS = config_manager.get_paths()
    COMMON_SETTINGS = config_manager.get_common_settings()
    SETTINGS = config_manager.get_settings()

    # Initialize path variables
    ENCODED_DIR = PATHS["encoded_dir"]
    ENCODING_LOG_FILE = PATHS["enc_log_path"]
    ENCODING_CSV_FILE = PATHS["enc_csv_path"]
    DECODING_LOG_FILE = PATHS["dec_log_path"]
    DECODING_CSV_FILE = PATHS["dec_csv_path"]
    QM_LOG_FILE = PATHS["qm_log_path"]

    # Get output directory from config, default to current directory
    OUTPUT_DIR = PATHS.get("metrics_dir", ".")
    PER_IMAGE_FILE = os.path.join(OUTPUT_DIR, "unified_per_image_results.csv")

    AV1_ENCODER_NAMES = list(COMMON_SETTINGS.get("av1", {}).keys())


def main() -> None:
    """Main function to orchestrate the image coding summary analysis."""
    # Parse command line arguments first
    args = setup_argument_parser()

    # Initialize configuration from the provided config file
    initialize_configuration(args.config_file)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize centralized data readers
    data_readers = DataReaders(PATHS, AV1_ENCODER_NAMES, COMMON_SETTINGS, BYTES_TO_MB)

    # Process data based on the selected branch (default is --read_qm)
    per_image_data = pd.DataFrame()

    if args.use_enc_qm_logs:
        per_image_data = process_enc_qm_logs(data_readers)

    elif args.use_per_image_log:
        per_image_data = process_per_image_logs(args.use_per_image_log)

    else:  # Default to --read_qm mode
        per_image_data = process_qm_directory(data_readers)

    # Save per-image data to CSV
    per_image_data.to_csv(PER_IMAGE_FILE, index=False)

    # Run analysis on the per-image data
    run_analysis(per_image_data, PER_IMAGE_FILE)


if __name__ == "__main__":
    main()
