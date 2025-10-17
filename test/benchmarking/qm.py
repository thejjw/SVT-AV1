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

import pathlib
import re
import subprocess
from logging import Logger
from typing import Dict, List, Optional, Tuple, Union

import utils


class QualityMetric:
    def __init__(self, ref_file: str, distorted_file: str):
        self.ref_file = ref_file
        self.distorted_file = distorted_file

    def calculate(self) -> Union[Optional[float], Optional[Dict[str, float]]]:
        raise NotImplementedError("Subclasses must implement calculate()")

    @classmethod
    def get_name(cls) -> str:
        raise NotImplementedError("Subclasses must implement get_name()")


class SSIMULACRA2Metric(QualityMetric):
    def __init__(
        self,
        ref_file: str,
        distorted_file: str,
        logger: Logger,
        ssimulacra2_bin: str,
        artifacts_path: str,
    ):
        super().__init__(ref_file, distorted_file)
        self.ssimulacra2_bin = ssimulacra2_bin
        self.artifacts_path = artifacts_path
        self.logger = logger

    def calculate(self) -> Optional[float]:
        command: List[str] = [self.ssimulacra2_bin, self.ref_file, self.distorted_file]
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        if result.returncode != 0:
            self.logger.error(
                f"Error calculating SSIMULACRA2: " f"{result.stderr.strip()}"
            )
            return None

        ref_stem = pathlib.Path(self.ref_file).stem
        out_path = os.path.join(self.artifacts_path, f"{ref_stem}.ssimulcra")
        with open(out_path, "w") as f:
            f.write(result.stdout)

        return float(result.stdout.strip())

    @classmethod
    def get_name(cls) -> str:
        return "SSIMULACRA2"


class MSSSIMMetric(QualityMetric):
    def __init__(
        self, source_file: str, test_file: str, logger: Logger, telephoto_bin: str
    ):
        super().__init__(source_file, test_file)
        self.telephoto_bin = telephoto_bin
        self.logger = logger

    def calculate(self) -> Optional[float]:
        try:
            command: List[str] = [
                self.telephoto_bin,
                "-in",
                self.ref_file,
                "-in",
                self.distorted_file,
                "-ms-ssim",
            ]

            result = subprocess.run(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

            if result.returncode != 0:
                self.logger.error(f"Error calculating MS-SSIM: {result.stderr.strip()}")
                return None

            # Parse MS-SSIM output from Telephoto
            re_mssim = re.compile(r"Computed value: ([0-9.]+)")
            match = re_mssim.search(result.stdout)
            if match:
                try:
                    score = float(match.group(1))
                    return score
                except ValueError:
                    self.logger.error(
                        f"Failed to convert MSSIM score to float: {match.group(1)}"
                    )
                    return None

            # Fallback: check stderr in case output is there
            match = re_mssim.search(result.stderr)
            if match:
                try:
                    score = float(match.group(1))
                    self.logger.debug(f"MSSIM score found in stderr: {score}")
                    return score
                except ValueError:
                    self.logger.error(
                        f"Failed to convert MSSIM score to float: {match.group(1)}"
                    )
                    return None

            self.logger.error(
                f"Could not parse MS-SSIM score from Telephoto output. STDOUT: '{result.stdout}' STDERR: '{result.stderr}'"
            )
            return None

        except Exception as e:
            self.logger.error(f"Error calculating MS-SSIM: {e}")
            return None

    @classmethod
    def get_name(cls) -> str:
        return "MS-SSIM"


class VMAFMetric(QualityMetric):
    def __init__(
        self,
        ref_file: str,
        distorted_file: str,
        logger: Logger,
        vmaf_bin: str,
        aom_ctc_model: str = "v6.0",
        artifacts_path: str = "",
    ):
        super().__init__(ref_file=ref_file, distorted_file=distorted_file)
        self.vmaf_bin = vmaf_bin
        self.aom_ctc_model = aom_ctc_model
        self.artifacts_path = artifacts_path
        self.logger = logger

    def calculate(self) -> Optional[Dict[str, float]]:
        ref_stem = pathlib.Path(self.ref_file).stem
        out_path = os.path.join(self.artifacts_path, f"{ref_stem}.xml")

        try:
            command: List[str] = [
                self.vmaf_bin,
                "-r",
                self.ref_file,
                "-d",
                self.distorted_file,
                "-o",
                out_path,
                "--xml",
                "--aom_ctc",
                self.aom_ctc_model,
            ]

            if self.ref_file.endswith(".yuv"):
                width, height, _ = utils.get_file_desc(os.path.basename(self.ref_file))
                command += [
                    "--width",
                    str(width),
                    "--height",
                    str(height),
                    "--pixel_format",
                    "420",
                    "--bitdepth",
                    "8",
                ]

            result = subprocess.run(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

            if result.returncode != 0:
                self.logger.error(f"Error calculating VMAF: {result.stderr.strip()}")
                return None

            import xml.etree.ElementTree as ET

            tree = ET.parse(out_path)
            vmaf_data = tree.getroot()

            metrics = {}

            # Find all metric elements in pooled_metrics section
            for metric_elem in vmaf_data.findall("./pooled_metrics/metric"):
                name = metric_elem.get("name")
                mean = float(metric_elem.get("mean"))
                metrics[name] = mean
            return metrics

        except Exception as e:
            self.logger.error(f"Error parsing VMAF output: {e}")
            return None

    @classmethod
    def get_name(cls) -> str:
        return "VMAF"


class QualityMetricsCalculator:
    def __init__(
        self,
        source_png_dir: str,
        source_y4m_dir: str,
        source_yuv_dir: str,
        ref_dir: str,
        y4m_ref_dir: str,
        yuv_ref_dir: str,
        logger: Logger,
        binaries: Dict[str, str],
        allow_metrics: Dict[str, bool],
        aom_ctc_model: str,
        artifacts_path: str = "",
    ):
        self.source_png_dir = source_png_dir
        self.ref_dir = ref_dir
        self.source_y4m_dir = source_y4m_dir
        self.y4m_ref_dir = y4m_ref_dir
        self.source_yuv_dir = source_yuv_dir
        self.yuv_ref_dir = yuv_ref_dir
        self.logger = logger
        self.binaries = binaries
        self.allow_metrics = allow_metrics
        self.aom_ctc_model = aom_ctc_model
        self.metrics: List = []
        self.artifacts_path = artifacts_path
        os.makedirs(self.artifacts_path, exist_ok=True)

        if allow_metrics.get("ssimulacra2", False):
            self.register_metric(
                lambda src, test: SSIMULACRA2Metric(
                    src,
                    test,
                    logger,
                    binaries["ssimulacra2"],
                    self.artifacts_path,
                )
            )

        if allow_metrics.get("vmaf", False):
            self.register_metric(
                lambda src, test: VMAFMetric(
                    src,
                    test,
                    logger,
                    binaries["vmaf"],
                    self.aom_ctc_model,
                    self.artifacts_path,
                )
            )

        if allow_metrics.get("mssim", False) and binaries.get("telephoto", None):
            self.register_metric(
                lambda src, test: MSSSIMMetric(src, test, logger, binaries["telephoto"])
            )

    def register_metric(self, metric_factory):
        self.metrics.append(metric_factory)

    def calculate_metrics(self, file_extension: str) -> Tuple[Dict[str, float], int]:
        total_values: Dict[str, float] = {}
        num_files: int = 0

        # Calculate SSIMULACRA2 metrics if PNG/PGM files are available
        if file_extension in [".png", ".pgm"] and self.allow_metrics.get(
            "ssimulacra2", False
        ):
            ssim_values, ssim_num_files = self.calculate(
                self.source_png_dir, self.ref_dir, file_extension, ["ssimulacra2"]
            )
            total_values.update(ssim_values)
            num_files = max(num_files, ssim_num_files)

        # Calculate MSSIM metrics if PNG/PGM files are available
        if file_extension in [".png", ".pgm"] and self.allow_metrics.get(
            "mssim", False
        ):
            mssim_values, mssim_num_files = self.calculate(
                self.source_png_dir, self.ref_dir, file_extension, ["ms-ssim"]
            )
            total_values.update(mssim_values)
            num_files = max(num_files, mssim_num_files)

        # Calculate VMAF metrics with Y4M files
        if self.allow_metrics.get("vmaf", False):
            vmaf_values, vmaf_num_files = self.calculate(
                self.source_y4m_dir, self.y4m_ref_dir, ".y4m", ["vmaf"]
            )
            total_values.update(vmaf_values)
            num_files = max(num_files, vmaf_num_files)

        return total_values, num_files

    def calculate(
        self,
        source_png_dir: str,
        ref_dir: str,
        file_extension: str,
        allowed_metric_names: List[str],
    ) -> Tuple[Dict[str, float], int]:
        comparison_files: List[str] = [
            f for f in os.listdir(ref_dir) if f.lower().endswith(file_extension)
        ]

        if not comparison_files:
            self.logger.warning(f"No {file_extension[1:]} files found in {ref_dir}.")
            return {}, 0

        total_values: Dict[str, float] = {}
        num_files: int = 0

        for file in comparison_files:
            source_file_path: str = os.path.join(source_png_dir, file)
            comparison_file_path: str = os.path.join(ref_dir, file)

            if not os.path.exists(source_file_path):
                self.logger.warning(
                    f"No matching {file_extension[1:]} file found in {source_png_dir}."
                )
                continue

            file_metrics = self.process_single_file(
                source_file_path, comparison_file_path, allowed_metric_names
            )

            # Update totals
            for metric_name, value in file_metrics.items():
                if metric_name not in total_values:
                    total_values[metric_name] = 0.0
                total_values[metric_name] += value

            # Log metrics for this file
            if file_metrics:
                log_msg = f"# {file}: "
                log_msg += ", ".join(
                    f"{name}={value:.4f}" for name, value in file_metrics.items()
                )
                log_msg += "\n" + "\n".join(
                    [
                        f"{comparison_file_path}, {metric_name}, {metric_value}"
                        for metric_name, metric_value in file_metrics.items()
                    ]
                )
                self.logger.info(log_msg)

            num_files += 1

        return total_values, num_files

    def process_single_file(
        self,
        source_file_path: str,
        comparison_file_path: str,
        allowed_metric_names: List[str],
    ) -> Dict[str, float]:
        file_metrics: Dict[str, float] = {}

        # Calculate each metric
        for metric_factory in self.metrics:
            metric_instance = metric_factory(source_file_path, comparison_file_path)

            # Only calculate if this metric is in the allowed list
            if metric_instance.get_name().lower() not in allowed_metric_names:
                continue

            value = metric_instance.calculate()

            if value is not None:
                metric_name = metric_instance.get_name()

                # Handle libvmaf's dictionary return type
                if isinstance(value, dict):
                    for sub_metric_name, sub_value in value.items():
                        if metric_name == "VMAF":
                            clean_metric_name = sub_metric_name
                        else:
                            clean_metric_name = f"{metric_name}_{sub_metric_name}"

                        file_metrics[clean_metric_name] = sub_value
                else:
                    # Handle SSIMULACRA2
                    file_metrics[metric_name] = value

        return file_metrics

    def calculate_single_file_metrics(
        self,
        ref_file: str,
        distorted_file: str,
        ref_y4m_file: str,
        distorted_y4m_file: str,
        ref_yuv_file: str,
        distorted_yuv_file: str,
    ) -> Dict[str, float]:
        """Calculate quality metrics for a single pair of files"""
        file_metrics: Dict[str, float] = {}

        # Calculate SSIMULACRA2 if PNG/PGM files are provided
        if (
            distorted_file
            and ref_file
            and any(distorted_file.lower().endswith(ext) for ext in [".png", ".pgm"])
            and self.allow_metrics.get("ssimulacra2", False)
        ):
            for metric_factory in self.metrics:
                metric_instance = metric_factory(ref_file, distorted_file)
                if metric_instance.get_name().lower() == "ssimulacra2":
                    value = metric_instance.calculate()
                    if value is not None:
                        file_metrics[metric_instance.get_name().lower()] = value

        # Calculate MSSIM if PNG files are provided
        if (
            distorted_file
            and ref_file
            and distorted_file.lower().endswith(".png")
            and self.allow_metrics.get("mssim", False)
        ):
            for metric_factory in self.metrics:
                metric_instance = metric_factory(ref_file, distorted_file)
                if metric_instance.get_name().lower() == "ms-ssim":
                    value = metric_instance.calculate()
                    if value is not None:
                        file_metrics["ms_ssim"] = value

        # Calculate VMAF if Y4M files are provided
        tmp_distorted_file = (
            distorted_y4m_file if distorted_y4m_file else distorted_yuv_file
        )
        tmp_ref_file = ref_y4m_file if ref_y4m_file else ref_yuv_file
        if (
            tmp_distorted_file
            and tmp_ref_file
            and os.path.exists(tmp_distorted_file)
            and os.path.exists(tmp_ref_file)
            and self.allow_metrics.get("vmaf", False)
        ):
            for metric_factory in self.metrics:
                metric_instance = metric_factory(tmp_ref_file, tmp_distorted_file)
                if metric_instance.get_name().lower() == "vmaf":
                    value = metric_instance.calculate()
                    if value is not None and isinstance(value, dict):
                        for sub_metric_name, sub_value in value.items():
                            file_metrics[sub_metric_name] = sub_value

        return file_metrics
