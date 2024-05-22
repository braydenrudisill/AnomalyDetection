import torch
from sklearn.metrics import auc
import numpy as np

from tqdm import tqdm

from modules.data import M10_SYNTHETIC_16K, PointCloudDataset, MVTEC_SCALING
from modules.pretrain_teacher import get_rf
from modules.models import TeacherNetwork, DecoderNetwork, KNNGraph


def main():
    anomaly_path = 'anomalies_train.txt'
    true_values_path = '../pivotdata/synthetic_bagel/test/0_gt.txt'

    with open(anomaly_path, 'r') as f:
        predicted_points = torch.tensor([list(map(float, line.split(' '))) for line in f])

    with open(true_values_path, 'r') as f:
        ground_truth_points = torch.tensor([list(map(float, line.split(' '))) for line in f]) * MVTEC_SCALING

    fprs = []
    pros = []

    resolution = 20
    # Check the PRO for many different thresholds and plot against FPR
    for i in tqdm(range(1*resolution, 3*resolution)):
        threshold = i / resolution
        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0
        for pred, actual in zip(predicted_points[:, 3], ground_truth_points[:, 3]):
            if pred > threshold and actual > 0:
                true_positives += 1
            if pred < threshold and actual == 0:
                true_negatives += 1
            if pred > threshold and actual == 0:
                false_positives += 1
            if pred < threshold and actual > 0:
                false_negatives += 1

        fpr = false_positives / (false_positives + true_negatives)
        pro = true_positives / (true_positives + false_negatives)

        fprs.append(fpr)
        pros.append(pro)

    fprs = [0] + fprs[::-1] + [1]
    pros = [0] + pros[::-1] + [1]
    print(fprs)
    print(pros)
    au_pro = trapezoid(fprs, pros, x_max=0.3)
    au_pro /= 0.3
    print(f"AU-PRO (FPR limit: {0.3}): {au_pro}")


def trapezoid(x, y, x_max=None):
    """
    This function calculates the definite integral of a curve given by
    x- and corresponding y-values. In contrast to, e.g., 'numpy.trapz()',
    this function allows to define an upper bound to the integration range by
    setting a value x_max.

    Points that do not have a finite x or y value will be ignored with a
    warning.

    Args:
        x:     Samples from the domain of the function to integrate
               Need to be sorted in ascending order. May contain the same value
               multiple times. In that case, the order of the corresponding
               y values will affect the integration with the trapezoidal rule.
        y:     Values of the function corresponding to x values.
        x_max: Upper limit of the integration. The y value at max_x will be
               determined by interpolating between its neighbors. Must not lie
               outside of the range of x.

    Returns:
        Area under the curve.
    """
    from bisect import bisect
    x = np.asarray(x)
    y = np.asarray(y)
    finite_mask = np.logical_and(np.isfinite(x), np.isfinite(y))
    if not finite_mask.all():
        print("""WARNING: Not all x and y values passed to trapezoid(...)
                 are finite. Will continue with only the finite values.""")
    x = x[finite_mask]
    y = y[finite_mask]

    # Introduce a correction term if max_x is not an element of x.
    correction = 0.
    if x_max is not None:
        if x_max not in x:
            # Get the insertion index that would keep x sorted after
            # np.insert(x, ins, x_max).
            ins = bisect(x, x_max)
            # x_max must be between the minimum and the maximum, so the
            # insertion_point cannot be zero or len(x).
            assert 0 < ins < len(x)

            # Calculate the correction term which is the integral between
            # the last x[ins-1] and x_max. Since we do not know the exact value
            # of y at x_max, we interpolate between y[ins] and y[ins-1].
            y_interp = y[ins - 1] + ((y[ins] - y[ins - 1]) *
                                     (x_max - x[ins - 1]) /
                                     (x[ins] - x[ins - 1]))
            correction = 0.5 * (y_interp + y[ins - 1]) * (x_max - x[ins - 1])

        # Cut off at x_max.
        mask = x <= x_max
        x = x[mask]
        y = y[mask]

    # Return area under the curve using the trapezoidal rule.
    return np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])) + correction


if __name__ == '__main__':
    main()
