import numpy as np
import torch.nn.functional as F
import torch.nn.functional as nnf
import torch
from torch import nn
import pystrum.pynd.ndutils as nd
import matplotlib.pyplot as plt
from scipy.spatial import distance

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals = []
        self.std = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.vals.append(val)
        self.std = np.std(self.vals)


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor).cuda()

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

class register_model(nn.Module):
    def __init__(self, img_size, mode='bilinear'):
        super(register_model, self).__init__()
        self.spatial_trans = SpatialTransformer(img_size, mode)

    def forward(self, x):
        if isinstance(x[1], np.ndarray):
            x[1] = torch.from_numpy(x[1])
        img = x[0].cuda()
        flow = x[1].cuda()
        out = self.spatial_trans(img, flow)
        return out


def dice_val_VOI(y_pred, y_true):
    '''Calculate the Dice coefficient of the predicted and true labels'''
    VOI_lbls = [1,2,3]

    pred = y_pred.detach().cpu().numpy()[0, 0, ...]
    true = y_true.detach().cpu().numpy()[0, 0, ...]
    DSCs = np.zeros((len(VOI_lbls), 1))
    idx = 0
    for i in VOI_lbls:
        pred_i = pred == i #创建二进制掩码，用于提取预测分割结果和真实标签中当前类别的像素
        true_i = true == i
        intersection = pred_i * true_i
        intersection = np.sum(intersection)
        union = np.sum(pred_i) + np.sum(true_i)
        dsc = (2.*intersection) / (union + 1e-5)
        DSCs[idx] =dsc
        idx += 1
    return np.mean(DSCs)

def hausdorff95(pred, true, spacing):
    '''
    Compute the 95th percentile Hausdorff Distance for a single pair of binary masks
    '''
    dist_matrix1 = distance.cdist(pred * spacing, true * spacing)
    dist_matrix2 = distance.cdist(true * spacing, pred * spacing)

    # Minimum distance for each point on pred to true and true to pred
    distances1 = np.min(dist_matrix1, axis=1)
    distances2 = np.min(dist_matrix2, axis=1)

    # Combine the distances and compute the 95th percentile
    all_distances = np.hstack([distances1, distances2])
    hd95 = np.percentile(all_distances, 95)
    return hd95

def compute_hd95_multilabel(y_pred, y_true, spacing=np.array([1, 1])):
    '''
    Compute the 95th percentile Hausdorff Distance (HD95) for multiple labels.
    '''
    pred = y_pred.detach().cpu().numpy()[0, 0, ...]
    true = y_true.detach().cpu().numpy()[0, 0, ...]
    VOI_lbls = [1, 3, 4, 5]
    HD95s = np.zeros(len(VOI_lbls))

    for idx, lbl in enumerate(VOI_lbls):
        pred_lbl = np.argwhere(pred == lbl)
        true_lbl = np.argwhere(true == lbl)
        if pred_lbl.size > 0 and true_lbl.size > 0:  # Both labels have points
            HD95s[idx] = hausdorff95(pred_lbl, true_lbl, spacing)
        else:
            HD95s[idx] = np.nan  # Assign NaN if one of the labels has no points to avoid zero division

    # Compute average HD95 excluding NaN values
    avg_hd95 = np.nanmean(HD95s)
    return avg_hd95


class SpatialTransform_flow(nn.Module):
    def __init__(self):
        super(SpatialTransform_flow, self).__init__()

    def forward(self, x, flow, sample_grid):
        sample_grid = sample_grid + flow
        size_tensor = sample_grid.size()
        sample_grid[0, :, :, 0] = (sample_grid[0, :, :, 0] - ((size_tensor[3] - 1) / 2)) / size_tensor[3] * 2
        sample_grid[0, :, :, 1] = (sample_grid[0, :, :, 1] - ((size_tensor[2] - 1) / 2)) / size_tensor[2] * 2
        flow = torch.nn.functional.grid_sample(x, sample_grid, mode='bilinear')
        return flow


def ssim(img1, img2, window_size=11, window_sigma=1.5, data_range=1.0):

    mu1 = F.conv2d(img1, torch.ones(1, 1, window_size, window_size).to(img1.device) / (window_size ** 2), padding=window_size // 2, groups=img1.shape[1])
    mu2 = F.conv2d(img2, torch.ones(1, 1, window_size, window_size).to(img1.device) / (window_size ** 2), padding=window_size // 2, groups=img1.shape[1])

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, torch.ones(1, 1, window_size, window_size).to(img1.device) / (window_size ** 2), padding=window_size // 2, groups=img1.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, torch.ones(1, 1, window_size, window_size).to(img1.device) / (window_size ** 2), padding=window_size // 2, groups=img1.shape[1]) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, torch.ones(1, 1, window_size, window_size).to(img1.device) / (window_size ** 2), padding=window_size // 2, groups=img1.shape[1]) - mu1_mu2

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

    return ssim_map.mean()

def jacobian_determinant_vxm(disp):
    """
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.
    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
              where vol_shape is of len nb_dims
    Returns:
        jacobian determinant (scalar)

    """

    # check inputs
    disp = disp.transpose(1, 2,0)
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    J = np.gradient(disp + grid)

    # 3D glow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else:  # must be 2

        dfdx = J[0]
        dfdy = J[1]

        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]

def DSC(pred, target):
    smooth = 1e-5
    m1 = pred.flatten()
    m2 = target.flatten()
    intersection = (m1 * m2).sum()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)



class ResizeTransform(nn.Module):
    """
    Resize a transform, which involves resizing the vector field *and* rescaling it.
    """

    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = 'linear'
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        # don't do anything if resize is 1
        return x


def plot_image(x1, x2, x3, x4, x5, x6):
    # Create a 2x3 grid of subplots
    fig, axs = plt.subplots(2, 3, figsize=(12, 8), dpi=600)  # 2 rows, 3 columns

    # Plot the first set of images
    axs[0, 0].imshow(x1, cmap='gray')
    axs[0, 0].set_title('Reference')
    axs[0, 0].axis('off')  # Hide axis

    axs[0, 1].imshow(x2, cmap='gray')
    axs[0, 1].set_title('Warped')
    axs[0, 1].axis('off')  # Hide axis

    axs[0, 2].imshow(x3, cmap='viridis', vmin=0, vmax=800)
    axs[0, 2].set_title('Diff.')
    axs[0, 2].axis('off')  # Hide axis

    # Plot the second set of images
    axs[1, 0].imshow(x4, cmap='gray')
    axs[1, 0].set_title('Reference')
    axs[1, 0].axis('off')  # Hide axis

    axs[1, 1].imshow(x5, cmap='gray')
    axs[1, 1].set_title('Warped')
    axs[1, 1].axis('off')  # Hide axis

    axs[1, 2].imshow(x6, cmap='viridis', vmin=0, vmax=800)
    axs[1, 2].set_title('Diff.')
    axs[1, 2].axis('off')  # Hide axis
    # Adjust layout for better readability
    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.show()



