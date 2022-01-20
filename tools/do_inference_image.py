import torch
from smoke.config import cfg
from smoke.utils.check_point import DetectronCheckpointer
from smoke.engine import (
    default_argument_parser,
    default_setup,
)
from smoke.data.datasets.evaluation.kitti.kitti_eval import generate_kitti_3d_detection
from smoke.modeling.detector import build_detection_model
from smoke.structures.params_3d import ParamsList
import numpy as np
import csv
import cv2
import os

def setup(args):
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    return cfg

base_path = '/home/ubuntu/Downloads/SMOKE/datasets/kitti/testing/'
image_path = base_path + 'image_2/'
calib_dir = base_path + 'calib/'

filename = '000003.png'

def main():
    cfg = setup(args)

    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    checkpointer = DetectronCheckpointer(
        cfg, model, save_dir=cfg.OUTPUT_DIR
    )
    ckpt = cfg.MODEL.WEIGHT if args.ckpt is None else args.ckpt
    _ = checkpointer.load(ckpt, use_latest=args.ckpt is None)

    model.eval()

    ori_image = cv2.imread(image_path + filename)
    image = cv2.resize(ori_image, (1280, 384), fx=None, fy=None, interpolation=cv2.INTER_NEAREST)

    # whether needs split image is or not, think again.
    image_b, image_g, image_r = cv2.split(image)
    image = cv2.merge([image_b, image_g, image_r])

    mean = cfg.INPUT.PIXEL_MEAN
    std = cfg.INPUT.PIXEL_STD

    image = image.astype(np.float32, copy=False)

    image /= 255.0
    image -= np.array(mean)
    image /= np.array(std)

    image = image.transpose(2, 0, 1).astype(np.float32)

    images = np.expand_dims(image, 0).repeat(1, axis=0)

    image_tensor = torch.from_numpy(images).to(device)

    # mean = cfg.INPUT.PIXEL_MEAN
    # std = cfg.INPUT.PIXEL_STD
    # image_tensor /= 255.0
    # image_tensor -= torch.tensor(mean)
    # image_tensor /= torch.tensor(std)
    # image_tensor = image_tensor.to(device)

    target = ParamsList(image_size=(1280, 384), is_train=False)

    trans_mat = torch.tensor([[2.5765e-01,  7.4013e-19,  1.4211e-14],
                              [1.2721e-17,  2.5765e-01, -3.0918e-01],
                              [0.0000e+00,  0.0000e+00,  1.0000e+00]])

    target.add_field("trans_mat", trans_mat)

    # get camera intrinsic matrix K
    file_name = filename[:-4] + ".txt"
    with open(os.path.join(calib_dir, file_name), 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=' ')
        for line, row in enumerate(reader):
            if row[0] == 'P2:':
                K = row[1:]
                K = [float(i) for i in K]
                K = np.array(K, dtype=np.float32).reshape(3, 4)
                K = K[:3, :3]
                break

    target.add_field("K", K)

    targets = [target]
    with torch.no_grad():
        output = model(image_tensor, targets)
        print(output)
        output_txt = "./tools/do_image_pred/" + filename[:-4] + ".txt"
        generate_kitti_3d_detection(output.cpu(), output_txt)


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    main()