import cv2
import numpy as np
from utils import data_utils
import torch
from models.modules.experimental import *
from utils.general import *
from utils.snake import snake_config


mean = np.array([0.224334, 0.224334, 0.224334],
                    dtype=np.float32).reshape(1, 1, 3)
std = np.array([0.201994, 0.201994, 0.201994],
               dtype=np.float32).reshape(1, 1, 3)
device = torch.device('cuda')

def demo(model,img_path):
    img = cv2.imread(img_path)
    ###
    # center = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    # scale = np.array([512, 512])
    # input_w, input_h = (512, 512)
    # trans_input = data_utils.get_affine_transform(center, scale, 0, [input_w, input_h])
    # inp = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)
    inp = cv2.resize(img,(512,512))
    orig_img = inp.copy()
    inp = (inp.astype(np.float32) / 255.)
    inp = inp.transpose(2, 0, 1)
    inp = torch.from_numpy(inp)
    if inp.ndimension() == 3: inp = inp.unsqueeze(0)
    inp = inp.to(device)
    ###

    model.eval()
    with torch.no_grad():
        out = model(inp)
        pred = out[0][0]
        pred = non_max_suppression(pred, 0.2, 0.3, classes=None,agnostic=False)
        for i, det in enumerate(pred):
            if det is not None and len(det):
                for *xyxy, conf, cls in det:
                    x_min = xyxy[0].cpu()
                    y_min = xyxy[1].cpu()
                    x_max = xyxy[2].cpu()
                    y_max = xyxy[3].cpu()
                    # score = conf.cpu()
                    # clas = cls.cpu()
                    cv2.rectangle(orig_img, (x_min, y_min), (x_max, y_max), 255, thickness=2)

        pyout = out[1]
        py = pyout['py'][-1].detach().cpu().numpy() * snake_config.down_ratio
        py = np.array(py).astype(int)
        orig_img = cv2.drawContours(orig_img, py, -1, (255, 255, 0), 2)

        cv2.imwrite('test.jpg', orig_img)


if __name__ == '__main__':
    pt_path = 'runs/train/exp/weights/last.pt'
    model = attempt_load(pt_path, map_location=device)  # load FP32 model
    img_path = 'tmp/leftImg8bit/train/eye/g0008.jpg'
    demo(model,img_path)