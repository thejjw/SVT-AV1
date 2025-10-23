# Copyright(c) 2025 Meta Platforms, Inc. and affiliates.
#
# This source code is subject to the terms of the BSD 2 Clause License and
# the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
# was not distributed with this source code in the LICENSE file, you can
# obtain it at https://www.aomedia.org/license/software-license. If the
# Alliance for Open Media Patent License 1.0 was not distributed with this
# source code in the PATENTS file, you can obtain it at
# https://www.aomedia.org/license/patent-license.

import math
import traceback
import typing
from operator import itemgetter

import numpy as np
import scipy.interpolate


def filter_rd(
    rate: typing.List[float], dist: typing.List[float], cutoff: float
) -> typing.Tuple[typing.List[float], typing.List[float]]:
    filtered_rate = []
    filtered_dist = []
    count = 0
    rate = np.sort(rate)
    dist = np.sort(dist)

    for i in range(len(rate)):
        if dist[i] <= cutoff or count < 4:
            filtered_rate.append(rate[i])
            filtered_dist.append(dist[i])
            count += 1

    return (filtered_rate, filtered_dist)


def bd_rate(
    rate1: typing.List[float],
    dist1: typing.List[float],
    rate2: typing.List[float],
    dist2: typing.List[float],
    piecewise: bool = True,
    filter: bool = False,
    threshold: int = 10000,
    integration_interval: typing.Optional[typing.Tuple[float, float]] = None,
) -> float:
    if filter:
        rate1, dist1 = filter_rd(rate1, dist1, threshold)
        rate2, dist2 = filter_rd(rate2, dist2, threshold)

    log_rate1 = np.log(rate1)
    log_rate2 = np.log(rate2)

    # rate method
    p1 = np.polyfit(dist1, log_rate1, 3)
    p2 = np.polyfit(dist2, log_rate2, 3)

    # integration interval
    min_int = max(min(dist1), min(dist2))
    max_int = min(max(dist1), max(dist2))
    if integration_interval:
        min_int = max(min_int, integration_interval[0])
        max_int = min(max_int, integration_interval[1])

    # find integral
    if not piecewise:
        p_int1 = np.polyint(p1)
        p_int2 = np.polyint(p2)

        int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
        int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)
    else:
        lin = np.linspace(min_int, max_int, num=100, retstep=True)
        interval = lin[1]
        samples = lin[0]
        v1 = scipy.interpolate.pchip_interpolate(
            np.sort(dist1), np.sort(log_rate1), samples
        )
        v2 = scipy.interpolate.pchip_interpolate(
            np.sort(dist2), np.sort(log_rate2), samples
        )
        # Calculate the integral using the trapezoid method on the samples.
        int1 = np.trapezoid(v1, dx=interval)
        int2 = np.trapezoid(v2, dx=interval)

    # find avg diff
    avg_exp_diff = (int2 - int1) / (max_int - min_int)
    avg_diff = (np.exp(avg_exp_diff) - 1) * 100
    return avg_diff


def non_decreasing(
    L: typing.List[float],
) -> bool:
    return all(x <= y for x, y in zip(L, L[1:]))


def check_monotonicity(
    RDPoints: typing.List[typing.Tuple[float, float]],
) -> bool:
    """
    check if the input list of RD points are monotonic, assuming the input
    has been sorted in the quality value non-decreasing order. expect the bit
    rate should also be in the non-decreasing order
    """
    br = [RDPoints[i][0] for i in range(len(RDPoints))]
    qty = [RDPoints[i][1] for i in range(len(RDPoints))]
    return non_decreasing(br) and non_decreasing(qty)


# BJONTEGAARD    Bjontegaard metric
# PCHIP method - Piecewise Cubic Hermite Interpolating Polynomial interpolation
def bd_rate_v2(
    qty_type: str,
    br1: typing.List[float],
    qtyMtrc1: typing.List[float],
    br2: typing.List[float],
    qtyMtrc2: typing.List[float],
) -> typing.Tuple[int, typing.Union[float, str]]:
    brqtypairs1 = []
    brqtypairs2 = []
    for i in range(min(len(qtyMtrc1), len(br1))):
        if br1[i] != "" and qtyMtrc1[i] != "":
            brqtypairs1.append((br1[i], qtyMtrc1[i]))
    for i in range(min(len(qtyMtrc2), len(br2))):
        if br2[i] != "" and qtyMtrc2[i] != "":
            brqtypairs2.append((br2[i], qtyMtrc2[i]))

    # sort the pair based on quality metric values in increasing order
    # if quality metric values are the same, then sort the bit rate in increasing order
    brqtypairs1.sort(key=itemgetter(1, 0))
    brqtypairs2.sort(key=itemgetter(1, 0))

    rd1_monotonic = check_monotonicity(brqtypairs1)
    rd2_monotonic = check_monotonicity(brqtypairs2)
    if rd1_monotonic is False or rd2_monotonic is False:
        # if rd1_monotonic is False:
        #    print(brqtypairs1)
        # if rd2_monotonic is False:
        #    print(brqtypairs2)
        # Take input from the user
        # input("Enter any key to continue: ")
        return (
            -1,
            f"Metric {qty_type}: Non-monotonic Error: {brqtypairs1} & {brqtypairs2}",
        )

    try:
        logbr1 = [math.log(x[0]) for x in brqtypairs1]
        qmetrics1 = [100.0 if x[1] == float("inf") else x[1] for x in brqtypairs1]
        logbr2 = [math.log(x[0]) for x in brqtypairs2]
        qmetrics2 = [100.0 if x[1] == float("inf") else x[1] for x in brqtypairs2]
    except ValueError:
        traceback.print_exc()
        return (-1, "Invalid Input")

    if not brqtypairs1 or not brqtypairs2:
        return (-1, "one of input lists is empty!")

    # remove duplicated quality metric value, the RD point with higher bit rate is removed
    dup_idx = [i for i in range(1, len(qmetrics1)) if qmetrics1[i - 1] == qmetrics1[i]]
    for idx in sorted(dup_idx, reverse=True):
        del qmetrics1[idx]
        del logbr1[idx]
    dup_idx = [i for i in range(1, len(qmetrics2)) if qmetrics2[i - 1] == qmetrics2[i]]
    for idx in sorted(dup_idx, reverse=True):
        del qmetrics2[idx]
        del logbr2[idx]

    # find max and min of quality metrics
    min_int = max(min(qmetrics1), min(qmetrics2))
    max_int = min(max(qmetrics1), max(qmetrics2))
    if min_int >= max_int:
        return (
            -1,
            f"Metric {qty_type} has no overlap from input 2 lists of quality metrics!: {qmetrics1} & {qmetrics2}",
        )

    # generate samples between max and min of quality metrics
    lin = np.linspace(min_int, max_int, num=100, retstep=True)
    interval = lin[1]
    samples = lin[0]

    # interpolation
    v1 = scipy.interpolate.pchip_interpolate(qmetrics1, logbr1, samples)
    v2 = scipy.interpolate.pchip_interpolate(qmetrics2, logbr2, samples)

    # Calculate the integral using the trapezoid method on the samples.
    int1 = np.trapezoid(v1, dx=interval)
    int2 = np.trapezoid(v2, dx=interval)

    # find avg diff
    avg_exp_diff = (int2 - int1) / (max_int - min_int)
    avg_diff = (math.exp(avg_exp_diff) - 1) * 100

    return (0, round(avg_diff, 4))


# calculation integration using staircase
def calc_staircase_integration(
    logbr: typing.List[float], qty: typing.List[float], min_qty: float, max_qty: float
) -> float:
    # first interpolate to generate a curve, x-axis is the quality and y-axis is the log bitrate
    lin = np.linspace(min_qty, max_qty, num=100, retstep=True)
    samples = lin[0]
    v = scipy.interpolate.pchip_interpolate(qty, logbr, samples)

    # filter out rd points that has quality out of the [min_qty, max_qty] range
    filtered_logbr = []
    filtered_qty = []
    for i in range(len(qty)):
        if (qty[i] <= max_qty) and (qty[i] >= min_qty):
            filtered_logbr.append(logbr[i])
            filtered_qty.append(qty[i])

    # add min_qty and max_qty points into the filtered list if they are missing
    if filtered_qty[0] > min_qty:
        filtered_qty.insert(0, min_qty)
        filtered_logbr.insert(0, v[0])

    if filtered_qty[-1] < max_qty:
        filtered_qty.append(max_qty)
        filtered_logbr.append(v[-1])

    # calculate the integral using stair case
    integal = 0
    prev_q = filtered_qty[0]
    for i in range(1, len(filtered_qty)):
        integal += (filtered_qty[i] - prev_q) * filtered_logbr[i]
        prev_q = filtered_qty[i]

    return integal


# BJONTEGAARD    Bjontegaard metric
# use stair case integal
def bd_rate_v3(
    qty_type: str,
    br1: typing.List[float],
    qtyMtrc1: typing.List[float],
    br2: typing.List[float],
    qtyMtrc2: typing.List[float],
) -> typing.Tuple[int, typing.Union[float, str]]:
    brqtypairs1 = []
    brqtypairs2 = []
    for i in range(min(len(qtyMtrc1), len(br1))):
        if br1[i] != "" and qtyMtrc1[i] != "":
            brqtypairs1.append((br1[i], qtyMtrc1[i]))
    for i in range(min(len(qtyMtrc2), len(br2))):
        if br2[i] != "" and qtyMtrc2[i] != "":
            brqtypairs2.append((br2[i], qtyMtrc2[i]))

    # sort the pair based on quality metric values in increasing order
    # if quality metric values are the same, then sort the bit rate in increasing order
    brqtypairs1.sort(key=itemgetter(1, 0))
    brqtypairs2.sort(key=itemgetter(1, 0))

    rd1_monotonic = check_monotonicity(brqtypairs1)
    rd2_monotonic = check_monotonicity(brqtypairs2)
    if rd1_monotonic is False or rd2_monotonic is False:
        # if rd1_monotonic is False:
        #    print(brqtypairs1)
        # if rd2_monotonic is False:
        #    print(brqtypairs2)
        # Take input from the user
        # input("Enter any key to continue: ")
        return (
            -1,
            f"Metric {qty_type}: Non-monotonic Error: {brqtypairs1} & {brqtypairs2}",
        )

    try:
        logbr1 = [math.log(x[0]) for x in brqtypairs1]
        qmetrics1 = [100.0 if x[1] == float("inf") else x[1] for x in brqtypairs1]
        logbr2 = [math.log(x[0]) for x in brqtypairs2]
        qmetrics2 = [100.0 if x[1] == float("inf") else x[1] for x in brqtypairs2]
    except ValueError:
        traceback.print_exc()
        return (-1, "Invalid Input")

    if not brqtypairs1 or not brqtypairs2:
        return (-1, "one of input lists is empty!")

    # remove duplicated quality metric value, the RD point with higher bit rate is removed
    dup_idx = [i for i in range(1, len(qmetrics1)) if qmetrics1[i - 1] == qmetrics1[i]]
    for idx in sorted(dup_idx, reverse=True):
        del qmetrics1[idx]
        del logbr1[idx]
    dup_idx = [i for i in range(1, len(qmetrics2)) if qmetrics2[i - 1] == qmetrics2[i]]
    for idx in sorted(dup_idx, reverse=True):
        del qmetrics2[idx]
        del logbr2[idx]

    # find max and min of quality metrics
    min_int = max(min(qmetrics1), min(qmetrics2))
    max_int = min(max(qmetrics1), max(qmetrics2))
    if min_int >= max_int:
        return (
            -1,
            f"Metric {qty_type} has no overlap from input 2 lists of quality metrics!: {qmetrics1} & {qmetrics2}",
        )

    int1 = calc_staircase_integration(logbr1, qmetrics1, min_int, max_int)
    int2 = calc_staircase_integration(logbr2, qmetrics2, min_int, max_int)

    # find avg diff
    avg_exp_diff = (int2 - int1) / (max_int - min_int)
    avg_diff = (math.exp(avg_exp_diff) - 1) * 100

    return (0, round(avg_diff, 4))
