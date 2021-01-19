import numpy as np
import torch


def norm_kernel(x, y):
    mean1 = torch.mean(x, dim=0)
    mean2 = torch.mean(y, dim=0)
    std1 = torch.std(x, dim=0)
    std2 = torch.std(y, dim=0)

    loss = torch.mean((mean1 - mean2) ** 2 + (std1 - std2) ** 2)

    return loss


def linear_kernel(x, y):
    s1 = torch.mean(torch.mm(x, x.t()))
    s2 = torch.mean(torch.mm(y, y.t()))
    s3 = torch.mean(torch.mm(x, y.t()))

    return s1 + s2 - s3


def quad_kernel(x, y):
    z = x.shape[1]
    assert y.shape[1] == z

    s1 = torch.mean((torch.mm(x, x.t())) ** 2)
    s2 = torch.mean((torch.mm(y, y.t())) ** 2)
    s3 = torch.mean((torch.mm(x, y.t())) ** 2)

    return (s1 + s2 - 2 * s3) / (2 * np.sqrt(z))


def gram_kernel(x, y):
    Nx = len(x)
    Ny = len(y)

    Gx = torch.mm(x.t(), x) / Nx
    Gy = torch.mm(y.t(), y) / Ny
    return torch.mean((Gx - Gy) ** 2)


def gaussian_kernel(x, y):
    x_sq_dist = torch.norm(x - x, dim=1) ** 2
    y_sq_dist = torch.norm(y - y, dim=1) ** 2
    xy_sq_dist = torch.norm(x - y, dim=1) ** 2

    sigma_sq = torch.mean(x_sq_dist + y_sq_dist + 2 * xy_sq_dist) / 4

    s1 = torch.mean(torch.exp(-x_sq_dist / (2 * sigma_sq)))
    s2 = torch.mean(torch.exp(-y_sq_dist / (2 * sigma_sq)))
    s3 = torch.mean(torch.exp(-xy_sq_dist / (2 * sigma_sq)))

    return s1 + s2 - 2 * s3

def frechet_kernel(act1, act2):
    from scipy.linalg import sqrtm
    act1, act2 = act1.numpy(), act2.numpy()
    
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2)

    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))

    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


kernel_map = {
    'quad': quad_kernel,
    'linear': linear_kernel,
    'gauss': gaussian_kernel,
    'norm': norm_kernel,
    'gram': gram_kernel,
    'frechet': frechet_kernel
}
