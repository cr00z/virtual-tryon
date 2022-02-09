# Main file from virtual_tryon project
# https://github.com/cr00z/virtual_tryon


from tqdm import tqdm
import warnings
from utils.convert import (
    process_image_bbox,
    convert_smpl_to_bbox,
    convert_bbox_to_img
)
from utils.video import (
    get_video,
    write_image,
    generate_video
)
from predictor import Predictor
from regressor import Regressor
from visualizer import Visualizer
from dress import get_garment_mesh


# configure path to input video file
input_path = './sample_data/my_sample.mp4'
# input_path = './sample_data/single_totalbody.mp4'
output_path = './output'
stop_frame = None # 10
verbose = False
garment_path = './Multi-Garment_dataset/125611504306885'
# garment_path = './Multi-Garment_dataset/125611487366942'
garment_type = 'Pants'
# garment_type = 'TShirtNoCoat'


if __name__ == '__main__':
    cap, cap_length = get_video(input_path)
    print(f'Video: {input_path}')
    print(f'Frames total: {cap_length}')

    predictor = Predictor()
    regressor = Regressor()
    visualizer = Visualizer()

    for current_frame in tqdm(
            range(cap_length), 'Processing: ', position=0, leave=True):

        _, img_original_bgr = cap.read()
        if current_frame == stop_frame or img_original_bgr is None:
            break

        if verbose:
            print(f'Frame {current_frame}')
            print('Predict')

        # supress internal warning for clean up output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predictor.predict(img_original_bgr)
        max_body_bbox = predictor.get_max_body_bbox()

        norm_img, box_scale_o2n, bbox_top_left = process_image_bbox(
            img_original_bgr, max_body_bbox
        )

        if verbose:
            print('Regress')
        pred_betas, pred_pose, pred_camera = regressor.regress(norm_img)

        if verbose:
            print('Get garment mesh')

        # for random pose and beta testing, use the options below:
        # betas = (np.random.rand(10) - 0.5) * 2.5
        # pose = np.random.rand(72) - 0.5

        garment_ret_posed = get_garment_mesh(
            garment_path, garment_type, pred_betas, pred_pose)

        if verbose:
            print('Convert to image')
        pred_camera = pred_camera.cpu().numpy().ravel()
        cam_scale = pred_camera[0]
        cam_trans = pred_camera[1:]
        pred_vertices_bbox = convert_smpl_to_bbox(
            garment_ret_posed.v, cam_scale, cam_trans
        )
        pred_vertices_img = convert_bbox_to_img(
            pred_vertices_bbox, box_scale_o2n, bbox_top_left
        )
        garment_ret_posed.v = pred_vertices_img

        if verbose:
            print('Visualize')
        res_image = visualizer.visualize(
            img_original_bgr, max_body_bbox, garment_ret_posed
        )

        write_image(output_path, current_frame, res_image)

        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.imshow(res_image[:, :, ::-1])
        plt.show()

    print('Generating video in out.mp4')
    generate_video(output_path)
    print('Done')
