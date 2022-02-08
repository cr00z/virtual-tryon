import numpy as np
import torch
import cv2


def get_transform(center, scale, res, rot=0):
    """Generate transformation matrix."""
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -res[1] / 2
        t_mat[1, 2] = -res[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t


def transform(pt, center, scale, res, invert=0, rot=0):
    """Transform pixel location to different reference."""
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int) + 1


def crop_bbox_info(img, center, scale, res=(224, 224)):
    """Crop image according to the supplied bounding box."""
    # Upper left point
    ul = np.array(transform([1, 1], center, scale, res, invert=1)) - 1
    # Bottom right point
    br = np.array(transform([res[0] + 1,
                             res[1] + 1], center, scale, res, invert=1)) - 1

    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    if new_shape[0] < 1 or new_shape[1] < 1:
        return None, None, None
    new_img = np.zeros(new_shape, dtype=np.uint8)

    # Compute bbox for Han's format
    bboxScale_o2n = res[0] / new_img.shape[0]  # 224/ 531

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])

    if new_y[0] < 0 or new_y[1] < 0 or new_x[0] < 0 or new_x[1] < 0:
        return None, None, None

    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1],
                                                    old_x[0]:old_x[1]]

    bboxTopLeft_inOriginal = (ul[0], ul[1])

    if new_img.shape[0] < 20 or new_img.shape[1] < 20:
        return None, None, None

    new_img = cv2.resize(new_img, res)
    return new_img, bboxScale_o2n, np.array(bboxTopLeft_inOriginal)


def process_image_bbox(img, bbox_xywh, input_res=224):
    img = img[:, :, ::-1].copy()

    center = bbox_xywh[:2] + 0.5 * bbox_xywh[2:]
    scale = max(bbox_xywh[2:]) / 200.0 * 1.2  # magic people, woodoo people

    img, box_scale_o2n, bbox_top_left = crop_bbox_info(
        img, center, scale, (input_res, input_res))

    norm_img = (img.copy()).astype(np.float32) / 255.
    norm_img = torch.from_numpy(norm_img).permute(2, 0, 1)

    # normalize_transform = Normalize(mean=smpl.IMG_NORM_MEAN, std=smpl.IMG_NORM_STD)
    # norm_img = normalize_transform(norm_img)
    return norm_img, box_scale_o2n, bbox_top_left


def convert_smpl_to_bbox(data_3d, scale, trans, app_trans_first=False):
    data_3d = data_3d.copy()
    resnet_input_size_half = 224 * 0.5
    if app_trans_first:  # Hand model
        data_3d[:, 0:2] += trans
        data_3d *= scale  # apply scaling
    else:
        data_3d *= scale  # apply scaling
        data_3d[:, 0:2] += trans
    # 112 is originated from hrm's input size (224,24)
    data_3d *= resnet_input_size_half
    return data_3d


def convert_bbox_to_img(data_3d, box_scale_o2n, bbox_top_left):
    data_3d = data_3d.copy()
    resnet_input_size_half = 224 * 0.5

    data_3d /= box_scale_o2n

    if not isinstance(bbox_top_left, np.ndarray):
        assert isinstance(bbox_top_left, tuple)
        assert len(bbox_top_left) == 2
        bbox_top_left = np.array(bbox_top_left)

    data_3d[:, :2] += (bbox_top_left + resnet_input_size_half / box_scale_o2n)

    return data_3d
