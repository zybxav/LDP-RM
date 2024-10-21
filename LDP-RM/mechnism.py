import itertools
import math
import random

import numpy as np
import scipy.special
from numpy import linalg as LA
from scipy.special import comb
from svme import SVME


# for perturbation and aggregation of S, we will try SW, PW and harmony
# this is the perturb and aggregation function of SW
def sw(report, eps, s, randomized_bins=256, domain_bins=256):
    eps /= (len(s) ** 2)
    ee = np.exp(eps)
    w = ((eps * ee) - ee + 1) / (2 * ee * (ee - 1 - eps)) * 2  # w=2b
    p = ee / (w * ee + 1)
    q = 1 / (w * ee + 1)  # wp + q = 1
    report2 = np.zeros_like(report)
    res = np.zeros_like(s)
    length = len(s)
    for row in range(length):
        for j in range(length):
            l = s[row][j][1]
            h = s[row][j][0]
            for k in range(len(report)):
                element = report[k][row][j]
                samples = (element - l) / (h - l)
                randoms = np.random.uniform(0, 1)
                # report
                if randoms <= (q * samples):
                    report2[k][row][j] = randoms / q - w / 2
                elif randoms > q * samples + p * w:
                    report2[k][row][j] = (randoms - q * samples - p * w) / q + samples + w / 2
                elif randoms > (q * samples):
                    report2[k][row][j] = (randoms - q * samples) / p + samples - w / 2

            # report matrix
            m = randomized_bins
            n = domain_bins
            m_cell = (1 + w) / m
            n_cell = 1 / n

            transform = np.ones((m, n)) * q * m_cell
            for i in range(n):
                left_most_v = (i * n_cell)
                right_most_v = ((i + 1) * n_cell)

                ll_bound = int(left_most_v / m_cell)
                lr_bound = int((left_most_v + w) / m_cell)
                rl_bound = int(right_most_v / m_cell)
                rr_bound = int((right_most_v + w) / m_cell)

                ll_v = left_most_v - w / 2
                rl_v = right_most_v - w / 2
                l_p = ((ll_bound + 1) * m_cell - w / 2 - ll_v) * (p - q) + q * m_cell
                r_p = ((rl_bound + 1) * m_cell - w / 2 - rl_v) * (p - q) + q * m_cell
                if rl_bound > ll_bound:
                    transform[ll_bound, i] = (l_p - q * m_cell) * (
                            (ll_bound + 1) * m_cell - w / 2 - ll_v) / n_cell * 0.5 + q * m_cell
                    transform[ll_bound + 1, i] = p * m_cell - (p * m_cell - r_p) * (
                            rl_v - ((ll_bound + 1) * m_cell - w / 2)) / n_cell * 0.5
                else:
                    transform[ll_bound, i] = (l_p + r_p) / 2
                    transform[ll_bound + 1, i] = p * m_cell

                lr_v = left_most_v + w / 2
                rr_v = right_most_v + w / 2
                r_p = (rr_v - (rr_bound * m_cell - w / 2)) * (p - q) + q * m_cell
                l_p = (lr_v - (lr_bound * m_cell - w / 2)) * (p - q) + q * m_cell
                if rr_bound > lr_bound:
                    if rr_bound < m:
                        transform[rr_bound, i] = (r_p - q * m_cell) * (
                                rr_v - (rr_bound * m_cell - w / 2)) / n_cell * 0.5 + q * m_cell

                    transform[rr_bound - 1, i] = p * m_cell - (p * m_cell - l_p) * (
                            (rr_bound * m_cell - w / 2) - lr_v) / n_cell * 0.5

                    if rr_bound - 1 > ll_bound + 2:
                        transform[ll_bound + 2: rr_bound - 1, i] = p * m_cell
                else:
                    transform[rr_bound, i] = (l_p + r_p) / 2
                    transform[rr_bound - 1, i] = p * m_cell

            max_iteration = 10000
            likelihood_threshold = 1e-3
            ns_hist, ns_dist = np.histogram(report2[:, row, j], bins=randomized_bins, range=(-w / 2, 1 + w / 2))
            temp = EMS(n, ns_hist, transform, max_iteration, likelihood_threshold) * len(report)
            res_s = 0
            for index in range(len(temp)):
                res_s += temp[index] * ((ns_dist[index + 1] + ns_dist[index]) / 2 * (h - l) + l)
            res[row, j] = res_s
    # res is the sum of estimated distribution of S_ij,
    # report2 is the record of perturbed data, which would be used to be summed up to compare the EMS result
    return res / len(report)


# this is essential for recovering of SW
def EMS(n, ns_hist, transform, max_iteration, likelihood_threshold):
    # smoothing matrix
    smoothing_factor = 2
    binomial_tmp = [scipy.special.binom(smoothing_factor, k) for k in range(smoothing_factor + 1)]
    smoothing_matrix = np.zeros((n, n))
    central_idx = int(len(binomial_tmp) / 2)
    for i in range(int(smoothing_factor / 2)):
        smoothing_matrix[i, : central_idx + i + 1] = binomial_tmp[central_idx - i:]
    for i in range(int(smoothing_factor / 2), n - int(smoothing_factor / 2)):
        smoothing_matrix[i, i - central_idx: i + central_idx + 1] = binomial_tmp
    for i in range(n - int(smoothing_factor / 2), n):
        remain = n - i - 1
        smoothing_matrix[i, i - central_idx + 1:] = binomial_tmp[: central_idx + remain]
    row_sum = np.sum(smoothing_matrix, axis=1)
    smoothing_matrix = (smoothing_matrix.T / row_sum).T

    # EMS
    theta = np.ones(n) / float(n)
    theta_old = np.zeros(n)
    r = 0
    sample_size = sum(ns_hist)
    old_likelihood = 0

    while LA.norm(theta_old - theta, ord=1) > 1 / sample_size and r < max_iteration:
        theta_old = np.copy(theta)
        X_condition = np.matmul(transform, theta_old)

        TMP = transform.T / X_condition

        P = np.copy(np.matmul(TMP, ns_hist))
        P = P * theta_old

        theta = np.copy(P / sum(P))

        # Smoothing step
        theta = np.matmul(smoothing_matrix, theta)
        theta = theta / sum(theta)

        likelihood = np.inner(ns_hist, np.log(np.matmul(transform, theta)))
        improvement = likelihood - old_likelihood

        if r > 1 and abs(improvement) < likelihood_threshold:
            break

        old_likelihood = likelihood

        r += 1

    return theta
    # this is the end of SW


def generate_binary_random(pr, first, second):
    if random.random() <= pr:
        return first
    else:
        return second


# laplace with sampling
def laplace_sample(report, eps, s):
    if len(report) == 0:
        return np.zeros_like(s)
    d = len(s) ** 2
    length = len(s)
    s_range = np.zeros_like(s)
    s_min = np.zeros_like(s)
    res = np.zeros(d)
    for row in range(length):
        for col in range(length):
            l = s[row][col][1]
            h = s[row][col][0]
            s_range[row][col] = h - l
            s_min[row][col] = l
    for k in range(len(report)):
        j = random.randint(0, d - 1)
        row = j // length
        col = j % length
        element = report[k][row][col]
        noise_samples = element + np.random.laplace(0, s_range[row][col] / eps, 1)
        tp_star = [0 for _ in range(d)]
        tp_star[j] = noise_samples[0]
        res += tp_star
    res = np.reshape(res, (length, length))
    return res / len(report)


# laplace method
def laplace(report, eps, s):
    if len(report) == 0:
        return np.zeros_like(s)
    length = len(s)
    eps /= (length ** 2)
    s_range = np.zeros_like(s)
    res = np.zeros_like(s)
    for row in range(length):
        for col in range(length):
            l = s[row][col][1]
            h = s[row][col][0]
            s_range[row][col] = h - l
            for k in range(len(report)):
                element = report[k][row][col]
                noise_samples = element + np.random.laplace(0, s_range[row][col] / eps, 1)
                res[row][col] += noise_samples
    return res / len(report)


# PM_matrix
def pm(report, eps, s):
    if len(report) == 0:
        return np.zeros_like(s)
    length = len(s)
    d = length ** 2
    s_range = np.zeros_like(s)
    s_min = np.zeros_like(s)
    res = np.zeros_like(s)
    C = (math.exp(eps / 2) + 1) / (math.exp(eps / 2) - 1)
    for row in range(length):
        for col in range(length):
            l = s[row][col][1]
            h = s[row][col][0]
            s_range[row][col] = h - l
            s_min[row][col] = l
    for k in range(len(report)):
        sample_row = random.randint(0, length - 1)
        sample_col = random.randint(0, length - 1)
        element = report[k][sample_row][sample_col]
        if s_range[sample_row][sample_col]:
            sample = (element - s_min[sample_row][sample_col]) / s_range[sample_row][sample_col] * 2 - 1
        else:
            sample = 0
        l_ti = (C + 1) * sample / 2 - (C - 1) / 2
        r_ti = l_ti + C - 1
        if random.uniform(0, 1) < math.exp(eps / 2) / (math.exp(eps / 2) + 1):
            res[sample_row][sample_col] += random.uniform(l_ti, r_ti)
        else:
            sample = random.uniform(0, C + 1)
            if sample <= (l_ti + C):
                res[sample_row][sample_col] += sample - C
            else:
                res[sample_row][sample_col] += sample - 1
    res = (res * d / len(report) + 1) / 2 * s_range + s_min
    return res


def pm_list(report, eps, s):
    if len(report) == 0:
        return np.zeros_like(s)
    length = len(s)
    dimension = length
    s_range = np.zeros_like(s)
    s_min = np.zeros_like(s)
    res = np.zeros_like(s)
    C = (math.exp(eps / 2) + 1) / (math.exp(eps / 2) - 1)
    for i in range(length):
        l = s[i][1]
        h = s[i][0]
        s_range[i] = h - l
        s_min[i] = l
    for k in range(len(report)):
        sample_index = random.randint(0, length - 1)
        element = report[k][sample_index]
        if s_range[sample_index]:
            sample = (element - s_min[sample_index]) / s_range[sample_index] * 2 - 1
        else:
            sample = 0
        l_ti = (C + 1) * sample / 2 - (C - 1) / 2
        r_ti = l_ti + C - 1
        if random.uniform(0, 1) < math.exp(eps / 2) / (math.exp(eps / 2) + 1):
            res[sample_index] += random.uniform(l_ti, r_ti)
        else:
            sample = random.uniform(0, C + 1)
            if sample <= (l_ti + C):
                res[sample_index] += sample - C
            else:
                res[sample_index] += sample - 1
    res = (res * dimension / len(report) + 1) / 2 * s_range + s_min
    return res


def gt(report, eps, s):
    if len(report) == 0:
        return np.zeros_like(s)
    res = np.zeros_like(s)
    # res = sum(report)
    for k in range(len(report)):
        res += report[k]
    res = res / len(report)
    return res


# multiple PW
def pm_m(report, eps, s):
    if len(report) == 0:
        return np.zeros_like(s)
    length = len(s)
    d = length ** 2
    s_range = np.zeros_like(s)
    s_min = np.zeros_like(s)
    k = max(1, min(d, int(eps / 2.5)))
    t_star = np.zeros(d)
    for row in range(length):
        for col in range(length):
            l = s[row][col][1]
            h = s[row][col][0]
            s_range[row][col] = h - l
            s_min[row][col] = l
    for l in range(len(report)):
        samples = random.sample(list(range(0, d)), k)
        for j in samples:
            row = j // length
            col = j % length
            element = report[l][row][col]
            if s_range[row][col]:
                sample = (element - s_min[row][col]) / s_range[row][col] * 2 - 1
            else:
                sample = 0
            t_star[j] += (d / k) * pm_s(sample, eps / k)
    res = t_star.reshape(length, length)
    res = (res / len(report) + 1) / 2 * s_range + s_min
    return res


# single pm
def pm_s(tp, eps):
    C = (math.exp(eps / 2) + 1) / (math.exp(eps / 2) - 1)
    l_ti = (C + 1) * tp / 2 - (C - 1) / 2
    r_ti = l_ti + C - 1

    if random.uniform(0, 1) < math.exp(eps / 2) / (math.exp(eps / 2) + 1):
        return random.uniform(l_ti, r_ti)
    else:
        sample = random.uniform(0, C + 1)
        if sample <= (l_ti + C):
            return sample - C
        else:
            return sample - 1


# HM
def hm(report, eps, s):
    if eps > 0.61:
        alpha = 1 - math.exp(-1 * eps / 2)
    else:
        alpha = 0
    if random.random() < alpha:
        return pm(report, eps, s)
    else:
        return duchi_sample(report, eps, s)


def AMN(report, eps, s):
    if eps > 0.61:
        alpha = 1 - math.exp(-1 * eps / 2)
    else:
        alpha = 0
    if random.random() < alpha:
        return pm_list(report, eps, s)
    else:
        return duchi_sample_list(report, eps, s)


def AMN_matrix(report, eps, s):
    if eps > 0.61:
        if random.random() < 0.5:
            return pm(report, eps, s)
    return duchi_sample(report, eps, s)


# single Duchi
def duchi_sample(report, eps, s):
    if len(report) == 0:
        return np.zeros_like(s)
    length = len(s)
    d = length ** 2
    s_range = np.zeros_like(s)
    s_min = np.zeros_like(s)
    res = np.zeros_like(s)
    for row in range(length):
        for col in range(length):
            l = s[row][col][1]
            h = s[row][col][0]
            s_range[row][col] = h - l
            s_min[row][col] = l
    for k in range(len(report)):
        sample_row = random.randint(0, length - 1)
        sample_col = random.randint(0, length - 1)
        element = report[k][sample_row][sample_col]
        if s_range[sample_row][sample_col]:
            sample = (element - s_min[sample_row][sample_col]) / s_range[sample_row][sample_col] * 2 - 1
        else:
            sample = 0
        t_star = (math.exp(eps) + 1) / (math.exp(eps) - 1)
        prob = (math.exp(eps) - 1) * sample / (2 * math.exp(eps) + 2) + 1 / 2
        if generate_binary_random(prob, 1, 0) == 1:
            res[sample_row][sample_col] += t_star
        else:
            res[sample_row][sample_col] -= t_star
    res = (res * d / len(report) + 1) / 2 * s_range + s_min
    return res


def duchi_sample_list(report, eps, s):
    if len(report) == 0:
        return np.zeros_like(s)
    length = len(s)
    dimension = length
    s_range = np.zeros_like(s)
    s_min = np.zeros_like(s)
    res = np.zeros_like(s)
    for row in range(length):
        l = s[row][1]
        h = s[row][0]
        s_range[row] = h - l
        s_min[row] = l
    for k in range(len(report)):
        sample_row = random.randint(0, length - 1)
        element = report[k][sample_row]
        if s_range[sample_row]:
            sample = (element - s_min[sample_row]) / s_range[sample_row] * 2 - 1
        else:
            sample = 0
        t_star = (math.exp(eps) + 1) / (math.exp(eps) - 1)
        prob = (math.exp(eps) - 1) * sample / (2 * math.exp(eps) + 2) + 1 / 2
        if generate_binary_random(prob, 1, 0) == 1:
            res[sample_row] += t_star
        else:
            res[sample_row] -= t_star
    res = (res * dimension / len(report) + 1) / 2 * s_range + s_min
    return res


# unbiased method of sw for mean estimation
def sw_unb(report, eps, s):
    if len(report) == 0:
        return np.zeros_like(s)
    length = len(s)
    d = length ** 2
    ee = np.exp(eps)
    b = (eps * ee - ee + 1) / (ee * (ee - 1 - eps))
    w = b * 2  # w=2b
    p = w * ee / (w * ee + 1)
    s_range = np.zeros_like(s)
    s_min = np.zeros_like(s)
    res = np.zeros_like(s)
    for row in range(length):
        for col in range(length):
            l = s[row][col][1]
            h = s[row][col][0]
            s_range[row][col] = h - l
            s_min[row][col] = l
    for k in range(len(report)):
        sample_row = random.randint(0, length - 1)
        sample_col = random.randint(0, length - 1)
        element = report[k][sample_row][sample_col]
        sample = (element - s_min[sample_row][sample_col]) / s_range[sample_row][sample_col] * 2 - 1
        if random.uniform(0, 1) < p:
            res[sample_row][sample_col] += random.uniform(sample - b, sample + b)
        else:
            sample_ = random.uniform(0, 2)
            if sample_ <= (sample + 1):
                res[sample_row][sample_col] += sample_ - 1 - b
            else:
                res[sample_row][sample_col] += sample_ - 1 + b
    res = (res * (b * ee + 1) / (b * ee - b) * d / len(report) + 1) / 2 * s_range + s_min
    return res


# multiple Duchi
def duchi(report, eps, s):
    if len(report) == 0:
        return np.zeros_like(s)
    length = len(s)
    d = length ** 2
    s_range = np.zeros_like(s)
    s_min = np.zeros_like(s)
    res = np.zeros(d)
    for row in range(length):
        for col in range(length):
            l = s[row][col][1]
            h = s[row][col][0]
            s_range[row][col] = h - l
            s_min[row][col] = l

    if d % 2 != 0:
        C_d = 2 ** (d - 1) / comb(d - 1, int((d - 1) / 2))
    else:
        C_d = (2 ** (d - 1) + 0.5 * comb(d, int(d / 2))) / comb(d - 1, int(d / 2))

    B = C_d * (math.exp(eps) + 1) / (math.exp(eps) - 1)
    neg_B = (-1) * B
    pool = [i for i in itertools.product([neg_B, B], repeat=d)]
    for k in range(len(report)):
        # print(k, row, col)
        samples = []
        for j in range(d):
            row = j // length
            col = j % length
            element = report[k][row][col]
            samples.append((element - s_min[row][col]) / s_range[row][col] * 2 - 1)  # 处理到(-1, 1)区间内
        v = [generate_binary_random(0.5 + 0.5 * samples[j], 1, -1) for j in range(d)]

        t_pos = []
        t_neg = []
        for t_star in pool:
            if np.dot(np.array(t_star), np.array(v)) > 0:
                t_pos.append(t_star)
            elif np.dot(np.array(t_star), np.array(v)) < 0:
                t_neg.append(t_star)
            else:
                t_pos.append(t_star)
                t_neg.append(t_star)
        if generate_binary_random(math.exp(eps) / (math.exp(eps) + 1), 1, 0) == 1:
            res += np.array(random.choice(t_pos))
        else:
            res += np.array(random.choice(t_neg))
    res = res.reshape(length, length)
    res = (res / len(report) + 1) / 2 * s_range + s_min
    return res


# harmony
def harmony(report, eps, s):
    if len(report) == 0:
        return np.zeros_like(s)
    length = len(s)
    d = length ** 2
    s_range = np.zeros_like(s)
    s_min = np.zeros_like(s)
    res = np.zeros(d)
    for row in range(length):
        for col in range(length):
            l = s[row][col][1]
            h = s[row][col][0]
            s_range[row][col] = h - l
            s_min[row][col] = l
    for k in range(len(report)):
        j = random.randint(0, d - 1)
        row = j // length
        col = j % length
        element = report[k][row][col]
        samples = (element - s_min[row][col]) / s_range[row][col] * 2 - 1
        tp_star = [0 for _ in range(d)]
        pr = (samples * (math.exp(eps) - 1) + math.exp(eps) + 1) / (2 * math.exp(eps) + 2)
        value = d * (math.exp(eps) + 1) / (math.exp(eps) - 1)
        if generate_binary_random(pr, 1, 0) == 1:
            tp_star[j] = value
        else:
            tp_star[j] = -1 * value
        res += tp_star
    res = np.reshape(res, (length, length))
    res = (res / len(report) * d + 1) / 2 * s_range + s_min
    return res


def harmony_m(report, eps, s):
    if len(report) == 0:
        return np.zeros_like(s)
    length = len(s)
    d = length ** 2
    s_range = np.zeros_like(s)
    s_min = np.zeros_like(s)
    res = np.zeros(d)
    for row in range(length):
        for col in range(length):
            l = s[row][col][1]
            h = s[row][col][0]
            s_range[row][col] = h - l
            s_min[row][col] = l
    for k in range(len(report)):
        j = random.randint(0, d - 1)
        row = j // length
        col = j % length
        element = report[k][row][col]
        samples = (element - s_min[row][col]) / s_range[row][col] * 2 - 1
        tp_star = [0 for _ in range(d)]
        pr = (samples * (math.exp(eps) - 1) + math.exp(eps) + 1) / (2 * math.exp(eps) + 2)
        value = d * (math.exp(eps) + 1) / (math.exp(eps) - 1)
        if generate_binary_random(pr, 1, 0) == 1:
            tp_star[j] = value
        else:
            tp_star[j] = -1 * value
        res += tp_star
    res = np.reshape(res, (length, length))
    res = (res / len(report) + 1) / 2 * s_range + s_min
    return res


def svme(report, eps, s):
    length = len(s)
    s_range = np.zeros_like(s)
    s_min = np.zeros_like(s)

    for row in range(length):
        l = s[row][1]
        h = s[row][0]
        s_range[row] = h - l
        s_min[row] = l

    args = {'client': len(report),
            'dim': len(s),
            'sparsity': len(s),
            'eps': eps,
            'neighbor_dist': -1,
            'delta': 0.000001, }
    svme_est = SVME(args)
    rr = []
    for r in report:
        rrr = []
        for i, v in enumerate(r):
            vv = (v - s_min[i]) / s_range[i] * 2 - 1
            rrr.append([i, vv])
        rr.append(rrr)

    svme_est.esitimate(rr)
    res = svme_est.get_mean()
    res = (res+1) / 2 * s_range + s_min
    return res