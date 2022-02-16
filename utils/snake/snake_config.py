import numpy as np
from config import cfg


mean = np.array([0.224334, 0.224334, 0.224334],
                    dtype=np.float32).reshape(1, 1, 3)
std = np.array([0.201994, 0.201994, 0.201994],
               dtype=np.float32).reshape(1, 1, 3)
data_rng = np.random.RandomState(123)
eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                   dtype=np.float32)
eig_vec = np.array([
    [-0.58752847, -0.69563484, 0.41340352],
    [-0.5832747, 0.00994535, -0.81221408],
    [-0.56089297, 0.71832671, 0.41158938]
], dtype=np.float32)

down_ratio = 4
scale = np.array([512, 512])
input_w, input_h = (512, 512)
scale_range = np.arange(1., 1., 1.)

voc_input_h, voc_input_w = (512, 512)
voc_scale_range = np.arange(1., 1., 1.)

box_center = False
center_scope = False

init = 'quadrangle'
init_poly_num = 20
poly_num = 80
gt_poly_num = 80
spline_num = 10

adj_num = 4

train_pred_box = False
box_iou = 0.7
confidence = 0.1
train_pred_box_only = True

train_pred_ex = False
train_nearest_gt = True

ct_score = cfg.ct_score

ro = 4

segm_or_bbox = 'segm'

