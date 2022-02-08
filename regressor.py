# Regress smpl parameters from image
# https://github.com/cr00z/virtual_tryon
# Use HMR pretrained model from frankmocap


import torch
from utils.geometry import rotation_matrix_to_angle_axis_torch
from hmr import hmr


smpl_mean_params = './assets/smpl_mean_params.npz'
regressor_checkpoint = './assets/2020_05_31-00_50_43-best-51.749683916568756.pt'


class Regressor:
    def __init__(self):
        self.device = torch.device('cuda')\
            if torch.cuda.is_available() else torch.device('cpu')
        self.model_regressor = hmr(smpl_mean_params).to(self.device)
        checkpoint = torch.load(regressor_checkpoint, map_location=self.device)
        self.model_regressor.load_state_dict(checkpoint['model'], strict=False)
        self.model_regressor.eval()

    def regress(self, norm_img):
        with torch.no_grad():
            # model forward
            pred_rotmat, pred_betas, pred_camera = self.model_regressor(
                norm_img[None].to(self.device))
            # pred_rotmat = torch.Size([1, 24, 3, 3])
            # pred_betas = torch.Size([1, 10])
            # pred_camera = torch.Size([1, 3])

        pred_pose = rotation_matrix_to_angle_axis_torch(
            pred_rotmat).to(self.device)
        pred_pose = pred_pose.reshape(pred_pose.shape[0], 72)
        # pred_pose = torch.Size([1, 72])

        return pred_betas, pred_pose, pred_camera
