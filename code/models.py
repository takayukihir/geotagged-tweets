import numpy as np
import scipy.stats
import scipy.optimize

# distance independent binomial model
def binomial_mle(num_target, num_all):
    '''Maximum likelihood estimation of the binomial model'''
    def negative_log_likelihood(p, data, num_all):
        pmf = scipy.stats.binom.pmf(data, num_all, p)
        pmf = np.where(pmf > 1e-12, pmf, 1e-12)
        return -np.sum(np.log(pmf))
    
    p = np.sum(num_target) / np.sum(num_all)
    aic = 2 + 2 * negative_log_likelihood(p, num_target, num_all)
    bic = np.log(len(num_target)) + 2 * negative_log_likelihood(p, num_target, num_all)
    return p, aic, bic

# distance dependent model

def core_periphery_model(distance, near_ratio, radius, exponent):
    '''segmented power law model'''
    return np.where(distance < radius, near_ratio, 
                    near_ratio / np.power(np.maximum(distance / radius, 1), exponent))

def core_periphery_mle_std(num_target, num_all, dist, params):
    q, r, a = params
    p = core_periphery_model(dist, q, r, a)
    dpdr = np.where(dist < r, 0, q * (a / r) * (dist / r)**(-a))
    dpdrr = np.where(dist < r, 0, q * (a * (a - 1) / r**2) * (dist / r)**(-a))
    dLdp = (num_target / p + (num_all - num_target) / (1 - p))
    dLdpp = -(num_target / p**2 + (num_all - num_target) / (1 - p)**2)
    dLdrr = dLdpp * dpdr**2 + dLdp * dpdrr
    fisher_information = - np.sum(dLdrr)
    return 1 / np.sqrt(fisher_information)

# maximum likelihood estimation of the density dependent model
def core_periphery_mle(num_target, num_all, dist):
    def negative_log_likelihood(params, data, num_all, dist):
        pmf = scipy.stats.binom.pmf(data, num_all, core_periphery_model(dist, *params))
        pmf = np.where(pmf > 1e-12, pmf, 1e-12)
        return -np.sum(np.log(pmf))
        # return -np.sum(scipy.stats.binom.logpmf(data, num_all, core_periphery_model(dist, *params))
    
    # initial guess
    near_ratio = 0.01
    exponent = 2
    bounds = [(0, 1), (0.01, None), (0.5, None)]
    
    # minimize negative log likelihood
    res = None
    for radius in [10, 20, 40, 80]:
        params0 = [near_ratio, radius, exponent]
        now = scipy.optimize.minimize(negative_log_likelihood, params0, 
                                      args=(num_target, num_all, dist),
                                      bounds=bounds, method='Nelder-Mead')
        if now.success and (res is None or now.fun < res.fun):
            res = now
    
    aic = 2 * len(params0) + 2 * res.fun
    bic = len(params0) * np.log(len(num_target)) + 2 * res.fun
    return res.x, aic, bic

# Functions for plotting contours

def bins_to_mesh(bins, log_scale=False):
    pos = [np.exp((np.log(b[:-1]) + np.log(b[1:])) * 0.5) if log_scale 
           else (b[:-1] + b[1:]) * 0.5 for b in bins]
    mesh = np.meshgrid(*pos)
    return mesh

def data_histogram(data_x, data_y, bins, kde=True):
    counts, _, _ = np.histogram2d(data_x, data_y, bins=bins, density=False)
    xmesh, ymesh = bins_to_mesh(bins)
    if kde:
        vals = np.vstack([data_x, data_y])
        kernel = scipy.stats.gaussian_kde(vals, bw_method='scott')
        grid = np.vstack([xmesh.ravel(), ymesh.ravel()])
        densities = np.reshape(kernel(grid), xmesh.shape)
    else:
        densities, _, _ = np.histogram2d(data_x, data_y, bins=bins, density=True)
        densities = densities.T
    return xmesh, ymesh, counts.T, densities

def random_data_from_model(num_all, p, rng, num_repeat=10):
    '''Generate random data from the binomial model'''
    longer_num_all = np.repeat(num_all, num_repeat)
    if isinstance(p, float):
        longer_p = p
    else:
        longer_p = np.repeat(p, num_repeat)
    return longer_num_all, rng.binomial(longer_num_all, longer_p)

def histogram_to_density_levels(counts, densities, fractions):
    '''Identify the density levels above which the given fractions of the data lie'''
    densities_sorted, counts_sorted = zip(*sorted(
        list(zip(densities.flatten(), counts.flatten())), reverse=True))
    dlevels = np.interp(np.sum(counts) * fractions, np.cumsum(counts_sorted), densities_sorted)
    return dlevels

def data_contours(data_x, data_y, nlevels=3, log_scale=False):
    if log_scale:
        data_x, data_y = zip(*[(x, y) for x, y in zip(data_x, data_y) if x > 0 and y > 0])
        data_x = np.log(data_x)
        data_y = np.log(data_y)
    
    bins = []
    for data in [data_x, data_y]:
        data_min, data_max = np.min(data), np.max(data)
        data_diff = data_max - data_min
        bins.append(np.linspace(data_min - data_diff * 1E-2, data_max + data_diff * 1E-2, 101))

    xmesh, ymesh, counts, densities = data_histogram(data_x, data_y, bins)
    fractions = 1 - np.geomspace(1E-3, 1E-1, nlevels)
    levels = histogram_to_density_levels(counts, densities, fractions)

    if log_scale:
        xmesh = np.exp(xmesh)
        ymesh = np.exp(ymesh)
    return xmesh, ymesh, densities, levels