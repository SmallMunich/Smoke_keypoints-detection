import os
import csv
import logging
import subprocess

from smoke.utils.miscellaneous import mkdir

ID_TYPE_CONVERSION = {
    0: 'Car',
    1: 'Cyclist',
    2: 'Pedestrian'
}


def kitti_evaluation(
        eval_type,
        dataset,
        predictions,
        output_folder,
):
    logger = logging.getLogger(__name__)
    if "detection" in eval_type:
        logger.info("performing kitti detection evaluation: ")
        do_kitti_detection_evaluation(
            dataset=dataset,
            predictions=predictions,
            output_folder=output_folder,
            logger=logger
        )


def do_kitti_detection_evaluation(dataset,
                                  predictions,
                                  output_folder,
                                  logger
                                  ):
    predict_folder = os.path.join(output_folder, 'data')  # only recognize data
    mkdir(predict_folder)

    for image_id, prediction in predictions.items():
        predict_txt = image_id + '.txt'
        predict_txt = os.path.join(predict_folder, predict_txt)

        generate_kitti_3d_detection(prediction, predict_txt)

    logger.info("Evaluate on KITTI dataset")
    output_dir = os.path.abspath(output_folder)
    os.chdir('tools/evaluation/kitti/')
    if not os.path.isfile('kitti_eval'):
        subprocess.Popen('g++ -O3 -DNDEBUG -o kitti_eval evaluate_object_3d_offline.cpp', shell=True)
    label_dir = getattr(dataset, 'label_dir')
    command = "./kitti_eval {} {}".format(label_dir, output_dir)
    output = subprocess.check_output(command, shell=True, universal_newlines=True).strip()
    logger.info(output)


def generate_kitti_3d_detection(prediction, predict_txt):
    with open(predict_txt, 'w', newline='') as f:
        w = csv.writer(f, delimiter=' ', lineterminator='\n')
        if len(prediction) == 0:
            w.writerow([])
        else:
            for p in prediction:
                p = p.numpy()
                p = p.round(4)
                type = ID_TYPE_CONVERSION[int(p[0])]
                row = [type, 0, 0] + p[1:].tolist()
                w.writerow(row)

    check_last_line_break(predict_txt)


def check_last_line_break(predict_txt):
    f = open(predict_txt, 'rb+')
    try:
        f.seek(-1, os.SEEK_END)
    except:
        pass
    else:
        if f.__next__() == b'\n':
            f.seek(-1, os.SEEK_END)
            f.truncate()
    f.close()