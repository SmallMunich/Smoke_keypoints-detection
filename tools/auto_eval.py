import argparse
import os
import subprocess
import matplotlib.pyplot as plt

def find_idx(line):
    return line.find(':')

def draw(x_ori, y, title, save_dir):
    x = range(len(x_ori))
    middle = [eval(i[1]) for i in y]
    large = [eval(i[0]) for i in y]
    small = [eval(i[2]) for i in y]
    plt.figure()
    plt.title(title)
    plt.plot(x, middle, label='middle')
    plt.plot(x, large, label='large')
    plt.plot(x, small, label='small')
    max_middle = middle.index(max(middle))
    plt.text(x[max_middle], middle[max_middle], "epoch: {}: {:.4f}".format(x_ori[max_middle], middle[max_middle]),
             ha='center', va='bottom', fontsize=10)
    plt.plot(x[max_middle], middle[max_middle], marker='o', markerfacecolor='red')
    plt.legend()
    plt.savefig(os.path.join(save_dir, "{}.png".format(title)))
    print("Saved "+os.path.join(save_dir, "{}.png".format(title)))
    # plt.show()

def get_args():
    parser = argparse.ArgumentParser('Auto inference all pytorch module on Kitti evaluation datasets.\n'
                                     '>> cd smoke \n'
                                     '>> python tools/auto_eval.py module_dir --gt_dir path/to/gt_dir')
    parser.add_argument('--module_dir', help='path to your module folder.', default='./tools/logs/')
    parser.add_argument('--gt_dir', default='datasets/kitti/training/label_2/', help='path to ground truth folder.')
    return parser.parse_args()

def main():
    args = get_args()
    gt_dir = args.gt_dir
    module_dir = args.module_dir

    command_infer_0 = 'python ./tools/plain_train_net.py --eval-only True --ckpt {} OUTPUT_DIR {}'
    command_eval_0  = 'tools/evaluation/kitti/kitti_eval {} {}'
    log_txt = os.path.join(module_dir, 'auto_eval_log.txt')

    all_module_names = [os.path.splitext(name)[0] for name in os.listdir(module_dir) if os.path.splitext(name)[1] =='.pth']
    all_module_names = sorted(all_module_names)

    if not os.path.exists(log_txt):
        for module_name in all_module_names:
            module_path = os.path.join(module_dir, '{}.pth'.format(module_name))
            command_infer_1 = command_infer_0.format(module_path, os.path.join(module_dir, module_name))
            print("Doing: ", command_infer_1)
            output_str1 = subprocess.check_output(command_infer_1, shell=True, universal_newlines=True).strip()
            print(output_str1)
        with open(log_txt, 'w') as f:
            for module_name in all_module_names:
                command_eval_1 = command_eval_0.format(gt_dir, os.path.join(module_dir, module_name, 'inference/kitti_train/data/'))
                print("Doing: ", command_eval_1)
                f.write("***** command: " + command_eval_1 + "\n")
                output_str2 = subprocess.check_output(command_eval_1, shell=True, universal_newlines=True)
                # print(output_str2)
                f.write(output_str2)
            f.close()
    else:
        print("{} has already existed, reload and begin to draw metrics...".format(log_txt))
    # -------------draw metrics--------------------
    file_lines = open(log_txt, 'r').readlines()
    file_lines = [i.strip() for i in file_lines]

    det_ap = []
    ori_ap = []
    det_ground_ap = []
    det_3d_ap = []
    for i, line in enumerate(file_lines):
        if 'car_detection AP' in line:
            det_ap.append(line[find_idx(line) + 1:].split())
        if 'car_orientation AP' in line:
            ori_ap.append(line[find_idx(line) + 1:].split())
        if 'car_detection_ground AP' in line:
            det_ground_ap.append(line[find_idx(line) + 1:].split())
        if 'car_detection_3d AP' in line:
            det_3d_ap.append(line[find_idx(line) + 1:].split())

    print(len(all_module_names))
    print(len(det_3d_ap))
    if len(det_3d_ap) < len(all_module_names):
        det_3d_ap.append(det_3d_ap[-1])
        det_ap.append(det_ap[-1])
        det_ground_ap.append(det_ground_ap[-1])
        ori_ap.append(ori_ap[-1])
    draw(all_module_names, det_3d_ap, 'car_detection_3d AP', module_dir)
    draw(all_module_names, det_ap, 'car_detection AP', module_dir)
    draw(all_module_names, det_ground_ap, 'car_detection_ground AP', module_dir)
    draw(all_module_names, ori_ap, 'car_orientation AP', module_dir)

if __name__ == '__main__':
    main()
