import numpy as np
import argparse
import os,sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.gridspec import GridSpec
from PIL import Image

class detectionInfo(object):
    def __init__(self, line):
        self.name = line[0]

        self.truncation = float(line[1])
        self.occlusion = int(line[2])

        # local orientation = alpha + pi/2
        self.alpha = float(line[3])

        # in pixel coordinate
        self.xmin = float(line[4])
        self.ymin = float(line[5])
        self.xmax = float(line[6])
        self.ymax = float(line[7])

        # height, weigh, length in object coordinate, meter
        self.h = float(line[8])
        self.w = float(line[9])
        self.l = float(line[10])

        # x, y, z in camera coordinate, meter
        self.tx = float(line[11])
        self.ty = float(line[12])
        self.tz = float(line[13])

        # global orientation [-pi, pi]
        self.rot_global = float(line[14])

    def member_to_list(self):
        output_line = []
        for name, value in vars(self).items():
            output_line.append(value)
        return output_line

def compute_birdviewbox(line, shape, scale):
    npline = [np.float64(line[i]) for i in range(1, len(line))]
    h = npline[7] * scale
    w = npline[8] * scale
    l = npline[9] * scale
    x = npline[10] * scale
    y = npline[11] * scale
    z = npline[12] * scale
    rot_y = npline[13]

    R = np.array([[-np.cos(rot_y), np.sin(rot_y)],
                  [np.sin(rot_y), np.cos(rot_y)]])
    t = np.array([x, z]).reshape(1, 2).T

    x_corners = [0, l, l, 0]  # -l/2
    z_corners = [w, w, 0, 0]  # -w/2


    x_corners += -w / 2
    z_corners += -l / 2

    # bounding box in object coordinate
    corners_2D = np.array([x_corners, z_corners])
    # rotate
    corners_2D = R.dot(corners_2D)
    # translation
    corners_2D = t - corners_2D
    # in camera coordinate
    corners_2D[0] += int(shape/2)
    corners_2D = (corners_2D).astype(np.int16)
    corners_2D = corners_2D.T

    return np.vstack((corners_2D, corners_2D[0,:]))

def draw_birdeyes(ax2, line_gt, line_p, shape):
    # shape = 900
    scale = 15

    pred_corners_2d = compute_birdviewbox(line_p, shape, scale)
    gt_corners_2d = compute_birdviewbox(line_gt, shape, scale)

    codes = [Path.LINETO] * gt_corners_2d.shape[0]
    codes[0] = Path.MOVETO
    codes[-1] = Path.CLOSEPOLY
    pth = Path(gt_corners_2d, codes)
    p = patches.PathPatch(pth, fill=False, color='orange', label='ground truth')
    ax2.add_patch(p)

    codes = [Path.LINETO] * pred_corners_2d.shape[0]
    codes[0] = Path.MOVETO
    codes[-1] = Path.CLOSEPOLY
    pth = Path(pred_corners_2d, codes)
    p = patches.PathPatch(pth, fill=False, color='green', label='prediction')
    ax2.add_patch(p)

def compute_3Dbox(P2, line):
    obj = detectionInfo(line)
    # Draw 2D Bounding Box
    xmin = int(obj.xmin)
    xmax = int(obj.xmax)
    ymin = int(obj.ymin)
    ymax = int(obj.ymax)
    width = xmax - xmin
    height = ymax - ymin
    box_2d = patches.Rectangle((xmin, ymin), width, height, fill=False, color='red', linewidth='3')
    #ax.add_patch(box_2d)

    # Draw 3D Bounding Box
    R = np.array([[np.cos(obj.rot_global), 0, np.sin(obj.rot_global)],
                  [0, 1, 0],
                  [-np.sin(obj.rot_global), 0, np.cos(obj.rot_global)]])

    x_corners = [0, obj.l, obj.l, obj.l, obj.l, 0, 0, 0]  # -l/2
    y_corners = [0, 0, obj.h, obj.h, 0, 0, obj.h, obj.h]  # -h
    z_corners = [0, 0, 0, obj.w, obj.w, obj.w, obj.w, 0]  # -w/2

    x_corners = [i - obj.l / 2 for i in x_corners]
    y_corners = [i - obj.h for i in y_corners]
    z_corners = [i - obj.w / 2 for i in z_corners]

    corners_3D = np.array([x_corners, y_corners, z_corners])
    corners_3D = R.dot(corners_3D)
    corners_3D += np.array([obj.tx, obj.ty, obj.tz]).reshape((3, 1))

    corners_3D_1 = np.vstack((corners_3D, np.ones((corners_3D.shape[-1]))))
    corners_2D = P2.dot(corners_3D_1)
    corners_2D = corners_2D / corners_2D[2]
    corners_2D = corners_2D[:2]

    return corners_2D, box_2d

def draw_3Dbox(ax, P2, line, color):

    corners_2D, box_2d = compute_3Dbox(P2, line)

    # draw all lines through path
    # https://matplotlib.org/users/path_tutorial.html
    bb3d_lines_verts_idx = [0, 1, 2, 3, 4, 5, 6, 7, 0, 5, 4, 1, 2, 7, 6, 3]
    bb3d_on_2d_lines_verts = corners_2D[:, bb3d_lines_verts_idx]
    verts = bb3d_on_2d_lines_verts.T
    codes = [Path.LINETO] * verts.shape[0]
    codes[0] = Path.MOVETO
    # codes[-1] = Path.CLOSEPOLYq
    pth = Path(verts, codes)
    p = patches.PathPatch(pth, fill=False, color=color, linewidth=2)

    width = corners_2D[:, 3][0] - corners_2D[:, 1][0]
    height = corners_2D[:, 2][1] - corners_2D[:, 1][1]
    # put a mask on the front
    front_fill = patches.Rectangle((corners_2D[:, 1]), width, height, fill=True, color=color, alpha=0.4)
    ax.add_patch(p)
    ax.add_patch(front_fill)
    #ax.add_patch(box_2d)


def visualization(args, image_path, label_path, calib_path, pred_path,
                  dataset, VEHICLES):

    for index in range(len(dataset)):
        image_file = os.path.join(image_path, dataset[index]+ '.png')
        label_file = os.path.join(label_path, dataset[index] + '.txt')
        prediction_file = os.path.join(pred_path, dataset[index]+ '.txt')
        calibration_file = os.path.join(calib_path, dataset[index] + '.txt')
        for line in open(calibration_file):
            if 'P2' in line:
                P2 = line.split(' ')
                P2 = P2[:13]
                P2 = np.asarray([float(i) for i in P2[1:]])
                P2 = np.reshape(P2, (3, 4))


        fig = plt.figure(figsize=(20.00, 5.12), dpi=100)

        fig.tight_layout()
        gs = GridSpec(1, 4)
        gs.update(wspace=0)  # set the spacing between axes.

        ax = fig.add_subplot(gs[0, :3])
        ax2 = fig.add_subplot(gs[0, 3:])

        image = Image.open(image_file).convert('RGB')
        shape = 900
        birdimage = np.zeros((shape, shape, 3), np.uint8)

        # no grand truth
        if not os.path.isfile(label_file):
            print('no grand turth file ...')
            label_file = prediction_file

        with open(label_file) as f1, open(prediction_file) as f2:
            for line_gt, line_p in zip(f1, f2):
                line_gt = line_gt.strip().split(' ')
                line_p = line_p.strip().split(' ')
                if line_gt[0] == 'DontCare' or line_p[0] == 'DontCare':
                    continue
                truncated = np.abs(float(line_p[1]))
                occluded = np.abs(float(line_p[2]))
                trunc_level = 1 if args.a == 'training' else 255

            # truncated object in dataset is not observable
                if line_p[0] in VEHICLES and truncated < trunc_level:
                    color = 'green'
                    if line_p[0] == 'Cyclist':
                        color = 'yellow'
                    elif line_p[0] == 'Pedestrian':
                        color = 'cyan'
                    draw_3Dbox(ax, P2, line_p, color)
                    draw_birdeyes(ax2, line_gt, line_p, shape)

        # visualize 3D bounding box
        ax.imshow(image)
        ax.set_xticks([]) #remove axis value
        ax.set_yticks([])

        # plot camera view range
        x1 = np.linspace(0, shape / 2)
        x2 = np.linspace(shape / 2, shape)
        ax2.plot(x1, shape / 2 - x1, ls='--', color='grey', linewidth=1, alpha=0.5)
        ax2.plot(x2, x2 - shape / 2, ls='--', color='grey', linewidth=1, alpha=0.5)
        ax2.plot(shape / 2, 0, marker='+', markersize=16, markeredgecolor='red')

        # visualize bird eye view
        ax2.imshow(birdimage, origin='lower')
        ax2.set_xticks([])
        ax2.set_yticks([])
        # add legend
        handles, labels = ax2.get_legend_handles_labels()
        try:
            legend = ax2.legend([handles[0], handles[1]], [labels[0], labels[1]], loc='lower right',
                            fontsize='x-small', framealpha=0.2)
            for text in legend.get_texts():
                plt.setp(text, color='w')
        except:
            print('something is wrong')


        print(dataset[index])
        if args.save == False:
            plt.show()
        else:
            fig.savefig(os.path.join(args.path, dataset[index]), dpi=fig.dpi, bbox_inches='tight', pad_inches=0)

def main(args):
    base_path = '/home/ubuntu/Data/kitti/testing/'
    label_path = base_path + 'label_2/'
    image_path = base_path + 'image_2/'
    calib_path = base_path + 'calib/'
    pred_path = '../logs/inference/kitti_test/data/'
    dataset = [name.split('.')[0] for name in sorted(os.listdir(pred_path))]
    VEHICLES = 'Car,Pedestrian,Cyclist'
    # VEHICLES = 'Car'

    visualization(args, image_path, label_path, calib_path, pred_path,
                  dataset, VEHICLES)

# def main(args):
#     # config the image data sets
#     base_path = '../../datasets/kitti/hlh_vis_test/' 
#     image_path = base_path + 'image_2/'
#     calib_path = base_path + 'calib/'
#     label_path = base_path + 'label_2/'
#     # config prediction path
#     pred_path =  '../../logs/inference/kitti_test/data/'
#     dataset = [name.split('.')[0] for name in sorted(os.listdir(pred_path))]
#     #VEHICLES = 'Car,Pedestrian,Cyclist'
#     VEHICLES = 'Car'

#     visualization(args, image_path, label_path, calib_path, pred_path,
#                   dataset, VEHICLES)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize 3D bounding box on images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-a', '-dataset', type=str, default='training', help='training dataset or tracklet')
    parser.add_argument('-s', '--save', type=bool, default=False, help='Save Figure or not')
    parser.add_argument('-p', '--path', type=str, default='./result/', help='Output Image folder')
    args = parser.parse_args()
    if not os.path.exists(args.path):
        os.mkdir(args.path)

    main(args)

# could use ffmpeg to generate a video easily
# ffmpeg -framerate 30 -pattern_type glob -i '/tmp/result/*.png'  -q 1 -s 1280x720 output.mp4

 
