import os, utils
from torch.utils.data import DataLoader
import datasets
import numpy as np
import torch
import scipy.io as sio
import matplotlib.pyplot as plt
from natsort import natsorted
from models import Registration_CEST_NeTwork
import random
import sys
import datetime
from skimage.metrics import structural_similarity as ssim

def jacobian_determinant(flow):
    """
    Calculate the determinant of Jacobian matrix for a given deformation field.

    Parameters:
        flow (numpy.ndarray): Deformation field with shape (2, 128, 128).

    Returns:
        determinant (numpy.ndarray): Determinant of Jacobian matrix with shape (128, 128).
    """
    # Calculate gradients
    dx = np.gradient(flow[0], axis=1)
    dy = np.gradient(flow[1], axis=0)

    # Jacobian matrix
    jac_det = dx + dy

    return jac_det


def count_negative_jacobian_voxels(flow):
    """
    Count the number of voxels where the determinant of Jacobian matrix is non-positive.

    Parameters:
        flow (numpy.ndarray): Deformation field with shape (2, 128, 128).

    Returns:
        count (int): Number of voxels with non-positive Jacobian determinant.
    """
    determinant = jacobian_determinant(flow)
    count = np.sum(determinant <= 0)

    return count


def negative_jacobian_voxel_percentage(flow):
    """
    Calculate the percentage of voxels where the determinant of Jacobian matrix is non-positive.

    Parameters:
        flow (numpy.ndarray): Deformation field with shape (2, 128, 128).

    Returns:
        percentage (float): Percentage of voxels with non-positive Jacobian determinant.
    """
    total_voxels = np.prod(flow.shape[1:])
    negative_voxels = count_negative_jacobian_voxels(flow)
    percentage = (negative_voxels / total_voxels) * 100

    return percentage

def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True

same_seeds(24)

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

def main():

    test_dir = 'Data/Data_numrical_simulation/'
    model_idx = -1
    model_dir = 'checkpoints/'


    img_size = (128,128)
    model = Registration_CEST_NeTwork.bidirectional_average_net(img_size,channels=16)
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx], map_location='cuda:0')['state_dict']
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))

    model.load_state_dict(best_model)
    model.cuda()
    reg_model = utils.register_model(img_size, 'nearest')
    reg_model.cuda()
    test_set = datasets.PairedImageInferDataset(test_dir)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
    folder_path = f'result/sim'
    if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
    with (torch.no_grad()):
            ssim_values = []
            for i, data in enumerate(test_loader):
                    starttime = datetime.datetime.now()
                    model.eval()
                    data = [t.cuda() for t in data]
                    x = data[0]
                    y = data[1]
                    target = data[2]
                    batch_size, channels, height, width, freq = x.shape
                    fixed_image = y[:,:,:,:,1]


                    size = (1, 1, 128, 128, freq)
                    moved_image = torch.ones(size)
                    flow = torch.ones((1, 2, 128, 128, freq))
                    jac_det = np.ones((128, 128, freq))

                    for n in range(freq):
                        moving_image = x[:, :, :, :, n]
                        output = model(moving_image, fixed_image)
                        moved_image[:, :, :, :, n] = output[0]
                        flow[:, :, :, :, n]= output[4]
                        flow_jac = output[4].detach().cpu().numpy()[0, :, :, :]
                        jac_det[ :, :, n] = utils.jacobian_determinant_vxm(flow_jac)


                    endtime = datetime.datetime.now()
                    elapsed_time = (endtime - starttime).total_seconds()
                    print(f'Registration {i + 1} took {elapsed_time} seconds.')
                    moved_image_np = moved_image.squeeze().cpu().detach().numpy()
                    target_np = target.squeeze().cpu().detach().numpy()
                    x_np = x.squeeze().cpu().detach().numpy()
                    flow_np = flow.squeeze().cpu().detach().numpy()
                    ssim_after = ssim(moved_image_np, target_np, data_range=moved_image_np.max() - x_np.min())
                    ssim_values.append(ssim_after)
                    file_name = f'moved_image_{i}.mat'
                    file_path = os.path.join(folder_path, file_name)
                    sio.savemat(file_path, {'moved_image': moved_image_np,'flow':flow_np,'jac_det':jac_det})
                    # plot representative image
                    diff = np.abs(target_np - moved_image_np)
                    # utils.plot_image(target_np[:, :, 0], moved_image_np[:, :, 0], diff[:, :, 0],
                    #                  target_np[:, :, 26], moved_image_np[:, :, 26], diff[:, :, 26])


    mean_ssim = np.mean(ssim_after)
    print(f"result_save_path: {folder_path}/{file_name}")
    print(f'Mean SSIM: {mean_ssim}')



def mk_grid_img(grid_step, line_thickness=1, grid_sz=(128, 128)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j + line_thickness - 1] = 1
    for i in range(0, grid_img.shape[0], grid_step):
        grid_img[i + line_thickness - 1, :] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img


def comput_fig(img):
    img = img.detach().cpu().numpy()[:, 0, :, :]
    batch_size = img.shape[0]
    fig = plt.figure(figsize=(12, 12), dpi=180)
    for i in range(batch_size):
        for j in range(img.shape[1]):
            plt.subplot(batch_size, img.shape[1], i * img.shape[1] + j + 1)
            plt.axis('off')
            plt.imshow(img[i, :, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig

# Function to create the directory if it does not exist
def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


# Custom Logger class to redirect stdout to a file
class Logger(object):
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    main()