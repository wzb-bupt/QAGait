import os
import os.path as osp
import argparse
import cv2
import numpy as np
from copy import deepcopy
import time
from tqdm import tqdm
from multiprocessing import Pool

# Usage (e.g., CASIA-B):
# python wild_connect_area.py \
# --input_path <input path of original dataset> \
# --output_path <output path> \
# --drop_path <dropped frames>

# python -u wild_connect_area.py \
# --input_path /data/wangzengbin/CASIA-B/Silhouette/GaitDatasetB-silh \
# --output_path /data/wangzengbin/CASIA-B/Silhouette/GaitDatasetB-silh-ConnectArea \
# --drop_path /data/wangzengbin/CASIA-B/Silhouette/GaitDatasetB-silh-ConnectArea-drop \
# 2>&1 | tee CASIA_B_MaxConnectArea.log

MAX_RATIO_THRES = 0.95

def connect_area(sil):
    # assert(len(np.unique(np.asarray(sil))) == 2) # assert binary
    # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(sil, connectivity=8)
    num_labels, labels = cv2.connectedComponents(sil)
    max_area = 0
    max_idx = -1
    for i in range(1, num_labels):
        sub_area = (labels == i).sum()
        if sub_area > max_area:
            max_area = sub_area
            max_idx = i
    zero_mask = (labels != max_idx)
    new_sil = deepcopy(sil)
    new_sil[zero_mask] = 0

    total_area = (sil > 0).sum()
    max_ratio = (max_area*1.0) / (total_area+1e-9)
    return max_ratio, new_sil

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Connection Area for Wild Gait Dataset')
    parser.add_argument('--input_path', default='', type=str, help='Input Path')
    parser.add_argument('--output_path', default='', type=str, help='Output Path')
    parser.add_argument('--drop_path', default=None, type=str, help='Drop Path')
    args = parser.parse_args()

    src_dir = args.input_path
    des_dir = args.output_path
    drop_dir = args.drop_path
    if not osp.exists(des_dir):
        os.makedirs(des_dir)
    if not osp.exists(drop_dir):
        os.makedirs(drop_dir)

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
                    
                sil_name_list = sorted(os.listdir(view_dir))
                max_ratio_list = []
                new_sil_list = []

                start_time = time.time() # time

                for sil_name in sil_name_list:
                    sil_path = osp.join(view_dir, sil_name)
                    sil = cv2.imread(sil_path, 0)
                    _, binary_sil = cv2.threshold(sil, 127, 255, cv2.THRESH_BINARY)
                    #############################################################
                    # diff_sil = binary_sil - sil
                    # print('BinaryThreshold: ', sil.shape, np.unique(np.asarray(sil)), np.min(diff_sil), np.max(diff_sil))
                    #############################################################
                    max_ratio, new_sil = connect_area(binary_sil)
                    max_ratio_list.append(max_ratio)
                    new_sil_list.append(new_sil)
                max_ratio_list = np.asarray(max_ratio_list)
                # del sil, binary_sil, new_sil

                # end_time = time.time()
                # execution_time = end_time - start_time
                # print(f'Process each sequence for MaxConnect: {execution_time/len(sil_name_list)}s')

                qlf_index = np.where(max_ratio_list > MAX_RATIO_THRES)[0]
                qlf_flag = True
                if len(qlf_index) < 15:
                    qlf_index = np.argsort(max_ratio_list)[-15:]
                    qlf_flag = False
                for i in range(len(sil_name_list)):
                    if i in qlf_index:
                        sil_name = sil_name_list[i]
                        des_sil_path = osp.join(des_view_dir, sil_name)
                        cv2.imwrite(des_sil_path, new_sil_list[i])
                    else:
                        # for visualization analysis
                        sil_name = sil_name_list[i]
                        sil_path = osp.join(view_dir, sil_name)
                        sil = cv2.imread(sil_path, 0)
                        _, binary_sil = cv2.threshold(sil, 127, 255, cv2.THRESH_BINARY)
                        _, new_sil = connect_area(binary_sil)
                        drop_sil_name = 'drop_{}_{}_{}_{:.2f}.png'.format(_id, _type, _view, max_ratio_list[i])
                        drop_sil_path = osp.join(drop_dir, drop_sil_name)
                        cv2.imwrite(drop_sil_path, np.hstack((sil, binary_sil, new_sil)))
                
                #############################################################
                # for qualitative analysis
                if not qlf_flag:
                    print('qlf_index={}, qlf_ratio={}'.format(qlf_index, max_ratio_list[qlf_index]))
                if len(qlf_index) < len(sil_name_list):
                    print('ID={}, Type={}, View={}, Raw_Frame={}, Now_Frames={}, Drop_Frames={}, Drop_Ratio={:.2f}'.format( \
                            _id, _type, _view, len(sil_name_list), len(qlf_index), len(sil_name_list)-len(qlf_index), 1-(len(qlf_index)*1.0)/len(sil_name_list)))
                #############################################################

    start_time = time.time()

    id_list = sorted(os.listdir(src_dir))
    # # del id_list[4]
    # # for _id in id_list:
    # #     process_id(_id)
    # from multiprocessing import Pool
    pool = Pool()
    pool.map(process_id, id_list)
    pool.close()
    pool.join()

    end_time = time.time()
    execution_time = end_time - start_time
    print(f'Total time for MaxConnect: {execution_time}s')