import os
import os.path as osp
import argparse
import cv2
import math
from math import *
import numpy as np
import pickle
import time

ANGLE_THRES = 30  # save rotate angle > ANGLE_THRES for visualization

print('******* ANGLE_THRES={} *******'.format(ANGLE_THRES))

def cut_image_resize(img):
    T_H, T_W = 64, 64
    y = img.sum(axis=1)
    y_top = (y != 0).argmax(axis=0)
    y_btm = (y != 0).cumsum(axis=0).argmax(axis=0)
    img = img[y_top:y_btm + 1, :]
    _r = img.shape[1] / img.shape[0]
    _t_w = int(T_H * _r)
    img = cv2.resize(img, (_t_w, T_H), interpolation=cv2.INTER_CUBIC)
    sum_point = img.sum()
    sum_column = img.sum(axis=0).cumsum()
    x_center = -1
    for i in range(sum_column.size):
        if sum_column[i] > sum_point / 2:
            x_center = i
            break
    if x_center < 0:
        return None
    h_T_W = int(T_W / 2)
    left = x_center - h_T_W
    right = x_center + h_T_W
    if left <= 0 or right >= img.shape[1]:
        left += h_T_W
        right += h_T_W
        _ = np.zeros((img.shape[0], h_T_W))
        img = np.concatenate([_, img, _], axis=1)
    img = img[:, left:right]
    return img


def sils_align_minAreaRect(sil_seq, align_type='frame-level', disturb=0, seq_restrict_flg=False):
    '''align + cutresize'''
    sil_align_cutresize_seq = []
    angle_seq = []

    if disturb == 0:
        random_disturb = 0
    else:
        random_disturb = np.random.uniform(-disturb, disturb, 1)

    for img in sil_seq:
        # Contours detection
        try:
            _, contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # In OpenCV 2.4, cv2.findContours returns three values
        except:
            contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # In OpenCV 3.x and later versions, cv2.findContours returns two values
        # Find the smallest rectangle
        rect = cv2.minAreaRect(contours[0])  
        # Calculating the rotation Angle
        width, height = rect[1]  
        angle = rect[2]
        # Rotate the long side to be parallel to the Y-axis
        if width > height:
            angle -= 90
        # Save Sequence frame rotation angles for subsequent analysis
        angle_seq.append(angle + float(random_disturb))

    if align_type == 'frame-level':
        angle = angle_seq
    elif align_type == 'sequence-level':
        # seq_restrict_flg = True
        ######################################
        # Restrict left-align and right-align
        if seq_restrict_flg:
            pos_group = [a for a in angle_seq if a >= 0]
            neg_group = [a for a in angle_seq if a < 0]
            angle = [np.mean(pos_group) if a >= 0 else np.mean(neg_group) for a in angle_seq]
        ######################################
        else:
            angle_mean = np.mean(angle_seq)
            angle = [angle_mean for a in angle_seq]

    for i, img in enumerate(sil_seq):
        # Determine the new H and W after rotation
        sils_W = img.shape[1]
        sils_H = img.shape[0]
        sils_rotation_H = int(sils_W*fabs(sin(radians(angle[i])))+sils_H*fabs(cos(radians(angle[i]))))
        sils_rotation_W = int(sils_H*fabs(sin(radians(angle[i])))+sils_W*fabs(cos(radians(angle[i]))))
        # Getting the rotation matrix
        # rotation_matrix = cv2.getRotationMatrix2D(rect[0], angle[i], 1)
        rotation_matrix = cv2.getRotationMatrix2D((sils_H//2, sils_W//2), angle[i], 1)
        rotation_matrix[0, 2] += (sils_rotation_W - sils_W) / 2
        rotation_matrix[1, 2] += (sils_rotation_H - sils_H) / 2
        # Correction of rotation
        aligned_img = cv2.warpAffine(img, rotation_matrix, (sils_rotation_W, sils_rotation_H))
        # cut & resize to 64*64
        aligned_img_cutresize = cut_image_resize(aligned_img)
        # save final aligned sequence
        sil_align_cutresize_seq.append(aligned_img_cutresize)

    return sil_align_cutresize_seq, angle_seq


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Connection Area for Wild Gait Dataset')
    parser.add_argument('--input_path', default='', type=str, help='Input Path')
    parser.add_argument('--output_path', default='', type=str, help='Output Path')
    parser.add_argument('--align_results_path', default=None, type=str, help='Align Results Path')
    parser.add_argument('--align_type', default='frame-level', type=str, help='Align Type: frame-level or sequence-level')
    parser.add_argument('--disturb', default=0, type=int, help='Random disturb')
    parser.add_argument('--seq_restrict_flg', action='store_true', help='Sequence-level Align (group align or not)')
    args = parser.parse_args()

    src_dir = args.input_path
    des_dir = args.output_path
    align_results_dir = args.align_results_path
    align_type = args.align_type
    disturb = args.disturb
    seq_restrict_flg = args.seq_restrict_flg

    if not osp.exists(des_dir):
        os.makedirs(des_dir)
    if not osp.exists(align_results_dir):
        os.makedirs(align_results_dir)
    
    def process_id(_id):
        
        id_dir = os.path.join(src_dir, _id)
        type_list = sorted(os.listdir(id_dir))
        for _type in type_list:
            type_dir = os.path.join(src_dir, _id, _type)
            view_list = sorted(os.listdir(type_dir))
            for _view in view_list:
                view_dir = os.path.join(src_dir, _id, _type, _view)
                des_view_dir = view_dir.replace(src_dir, des_dir)
                if not osp.exists(des_view_dir):
                    os.makedirs(des_view_dir)
                
                sil_pkl = osp.join(view_dir, '{}.pkl'.format(_view))
                sil_seq = pickle.load(open(sil_pkl, 'rb'))
                num_sil = sil_seq.shape[0]
                
                # start_time = time.time()
                
                sil_align_seq, angle_seq = sils_align_minAreaRect(sil_seq, align_type, disturb, seq_restrict_flg)
                
                # end_time = time.time()
                # execution_time = end_time - start_time
                # print(f'Process each sequence for MaxConnect: {execution_time/num_sil}s')

                sil_align_pkl = osp.join(des_view_dir, '{}.pkl'.format(_view))
                pickle.dump(sil_align_seq, open(sil_align_pkl, 'wb'))
                # #####################################################
                # for visualization analysis
                angle_analisis_mean = sum([abs(x) for x in angle_seq]) / len(angle_seq)
                # print('ID={}, Type={}, View={}, angle_analisis_mean={}'.format(_id, _type, _view, angle_analisis_mean))

                if angle_analisis_mean > ANGLE_THRES:
                    # print(type(angle_analisis_mean))
                    # print(type(angle_seq[0]))
                    for i in range(num_sil):
                        large_angle_seq_name = 'visual_{}_{}_{}_{:.2f}_angle_mean_{:.2f}.png'.format(_id, _type, _view, angle_seq[i], angle_analisis_mean)
                        # large_angle_seq_name = 'visual_{}_{}_{}_angle_mean_{:.2f}.png'.format(_id, _type, _view, angle_analisis_mean[0])
                        sils_merge = np.concatenate([sil_seq[i], sil_align_seq[i]], 1)
                        large_angle_path = osp.join(align_results_dir, large_angle_seq_name)
                        
                        cv2.imwrite(large_angle_path, sils_merge)
                # #####################################################
                # for qualitative analysis
                if angle_analisis_mean > ANGLE_THRES:
                    print('ID={}, Type={}, View={}, Angle_analisis_mean={:.2f}'.format( \
                            _id, _type, _view, angle_analisis_mean))
                # #####################################################
        
    start_time = time.time()

    id_list = sorted(os.listdir(src_dir))
    # for _id in id_list:
    #     process_id(_id)
    from multiprocessing import Pool
    pool = Pool()
    pool.map(process_id, id_list)
    pool.close()
    pool.join()

    end_time = time.time()
    execution_time = end_time - start_time
    print(f'Total time for Alignment: {execution_time}s')