import logging
from tqdm import tqdm

import torch

from smoke.utils import comm
from smoke.utils.timer import Timer, get_time_str
from smoke.data.datasets.evaluation import evaluate
import cv2
import numpy as np
from smoke.structures.params_3d import ParamsList
import os
import csv

def test_unit(img_path='/home/ubuntu/Downloads/SMOKE/datasets/kitti/training/image_2/000010.png', device='cuda'):
    ori_image = cv2.imread(img_path)
    image = cv2.resize(ori_image, (384, 1280), fx=None, fy=None, interpolation=cv2.INTER_NEAREST)
    image = image.transpose(2, 0, 1).astype(np.float32)
    images = np.expand_dims(image, 0).repeat(1, axis=0)

    image_tensor = torch.from_numpy(images).to(device)
    target = ParamsList(image_size=(ori_image.shape[1], ori_image.shape[0]), is_train=False)

    trans_mat = torch.tensor([[2.5765e-01, -0.0000e+00, 2.5765e-01],
                              [-2.2884e-17, 2.5765e-01, -3.0918e-01],
                              [0, 0, 1]], )

    target.add_field("trans_mat", trans_mat)

    # get camera intrinsic matrix K
    calib_dir = "/home/ubuntu/Downloads/SMOKE/datasets/kitti/training/calib/"
    file_name = "000010.txt"
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

    return image_tensor, targets



def compute_on_dataset(model, data_loader, device, timer=None):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for _, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch["images"], batch["targets"], batch["img_ids"]

        images = images.to(device)
        with torch.no_grad():
            if timer:
                timer.tic()
            # import pdb; pdb.set_trace()
            # images_tensor, targets_tensor = test_unit()
            # output = model(images_tensor, targets_tensor)
            output = model(images, targets)
            if timer:
                torch.cuda.synchronize()
                timer.toc()
            output = output.to(cpu_device)
        results_dict.update(
            {img_id: output for img_id in image_ids}
        )
    return results_dict


def inference(
        model,
        data_loader,
        dataset_name,
        eval_types=("detections",),
        device="cuda",
        output_folder=None,
):
    device = torch.device(device)
    num_devices = comm.get_world_size()
    logger = logging.getLogger(__name__)
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))

    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions = compute_on_dataset(model, data_loader, device, inference_timer)
    comm.synchronize()

    # input_names = ['data']
    # output_names = ['prediction']
    # torch.onnx.export(model, input, "/home/ubuntu/Downloads/SMOKE/tools/smoke.onnx", verbose=False, input_names=input_names, output_names=output_names)
    # print("export onnx succ...")

    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )
    if not comm.is_main_process():
        print("comm.is_main_process() is false...")
        return
    print("comm.is_main_process() is true ---> ")
    return evaluate(eval_type=eval_types,
                    dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder, )
