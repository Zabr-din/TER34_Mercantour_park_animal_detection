from detectron2.engine import DefaultPredictor

import os
import pickle

from utils import *

cfg_path = "IS_cfg.pickle"

with open(cfg_path, "rb") as f:
	cfg = pickle.load(f)

# load the retrained weights
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

predictor = DefaultPredictor(cfg)

image_path = "image.jpg"

on_image(image_path, predictor)
