# Predict bbox in image
# https://github.com/cr00z/virtual_tryon


import numpy as np
import torch
import matplotlib.pyplot as plt
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer


detectron2_config = 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'
detectron2_weights = 'detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/' \
                     '137849458/model_final_280758.pkl'
person_class = 0


class Predictor:
    def __init__(self):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(detectron2_config))
        if not torch.cuda.is_available():
            self.cfg.MODEL.DEVICE = 'cpu'
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        self.cfg.MODEL.WEIGHTS = detectron2_weights
        self.predictor = DefaultPredictor(self.cfg)
        self.outputs = None

    def predict(self, img):
        self.outputs = self.predictor(img)
        return self.outputs

    def get_max_body_bbox(self):
        """ Sort the person bbox using size and get max """
        person_idxs = self.outputs['instances'].pred_classes == person_class
        person_bboxes = self.outputs['instances'][person_idxs].pred_boxes.tensor
        bboxes_size = [(c[2] - c[0]) * (c[3] - c[1]) for c in person_bboxes]
        max_body_bbox = person_bboxes[np.argmax(bboxes_size)]

        # convert to XYWH
        max_body_bbox[2] = max_body_bbox[2] - max_body_bbox[0]
        max_body_bbox[3] = max_body_bbox[3] - max_body_bbox[1]

        return max_body_bbox

    # TODO: remove
    def visualise(self, img):
        """ Method for visualize some test results """
        v = Visualizer(
            img[:, :, ::-1],
            MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
            scale=1.2
        )
        v = v.draw_instance_predictions(self.outputs['instances'].to('cpu'))
        plt.figure(figsize=(12, 8))
        plt.imshow(v.get_image()[:, :, ::-1])
        plt.show()
