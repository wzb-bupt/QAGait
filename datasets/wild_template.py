import os
import os.path as osp
import argparse
import cv2
import math
import numpy as np
import pickle
import time

np.seterr(divide='ignore',invalid='ignore')

# TEMPLATE_MATCH_THRES = 100

def cut_image_resize(img, T_W=44, T_H=64):
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


def template_match_dist_man_woman(sil_seq, template_seq):
    # assert(len(np.unique(np.asarray(sil_seq))) == 2) # assert binary

    MinTempMatchDist_results = []

    for img in sil_seq:
        # img_cutresize = cut_image_resize(img)

        dist_list = []
        for temp_img in template_seq:
            dist = cv2.matchShapes(img, temp_img, cv2.CONTOURS_MATCH_I2, 0)
            dist_list.append(dist)
        
        # 
        MinTempMatchDist = np.min(dist_list) * 10000  # results*10000 to make more intuitive 
        MinTempMatchDist_results.append(MinTempMatchDist)
    
    TempMatch_idx = np.where(np.asarray(MinTempMatchDist_results) < TEMPLATE_MATCH_THRES)[0]

    MinTempMatchDist_results = np.asarray(MinTempMatchDist_results)

    return MinTempMatchDist_results, TempMatch_idx


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Connection Area for Wild Gait Dataset')
    parser.add_argument('--input_path', default='', type=str, help='Input Path')
    parser.add_argument('--output_path', default='', type=str, help='Output Path')
    parser.add_argument('--template_path', default='', type=str, help='Template Path')
    parser.add_argument('--drop_path', default=None, type=str, help='Drop Path')
    # parser.add_argument('--drop_seq_frame_0_path', default=None, type=str, help='Drop Seq Frames = 0 Path')
    parser.add_argument('--threshold', default=100, type=int, help='Template Match Threshold')
    # parser.add_argument('--start_id', default=0, type=int, help='Gait3D_start_id')
    # parser.add_argument('--end_id', default=4000, type=int, help='Gait3D_end_id')
    parser.add_argument('--height_num', default=3, type=int, help='height_num')

    args = parser.parse_args()

    src_dir = args.input_path
    des_dir = args.output_path
    drop_dir = args.drop_path
    height_num = args.height_num
    # drop_seq_0_dir = args.drop_seq_frame_0_path
    TEMPLATE_MATCH_THRES = args.threshold
    # start_id = args.start_id
    # end_id = args.end_id
    template_dir = args.template_path
    if not osp.exists(des_dir):
        os.makedirs(des_dir)
    if not osp.exists(drop_dir):
        os.makedirs(drop_dir)
    # if not osp.exists(drop_seq_0_dir):
    #     os.makedirs(drop_seq_0_dir)

    # ############# Template Loader (man & woman  ===  height=0  ===) ###############
    if height_num == 1:
        template_man_woman_seq = []
        for gender in ['man', 'woman']:
            for _camera in sorted(os.listdir(osp.join(template_dir, gender))):
                camera_dir = osp.join(template_dir, gender, _camera)
                template_pkl = osp.join(camera_dir, '{}.pkl'.format(_camera))
                template_seq = pickle.load(open(template_pkl, 'rb'))
                # template_seq = [cut_image_resize(temp_img) for temp_img in template_seq]  # if 64*64
                template_man_woman_seq.append(template_seq)
        template_seq_total = np.concatenate(template_man_woman_seq, 0)  # Total = 868 frames

    # ############# Template Loader (man & woman  ===  height=0/1/2  ===) ###############
    if height_num == 3:
        template_man_woman_seq = []
        for gender in ['man', 'woman']:
            for height in sorted(os.listdir(osp.join(template_dir, gender))):
                for _camera in sorted(os.listdir(osp.join(template_dir, gender, height))):
                    camera_dir = osp.join(template_dir, gender, height, _camera)
                    template_pkl = osp.join(camera_dir, '{}.pkl'.format(_camera))
                    template_seq = pickle.load(open(template_pkl, 'rb'))
                    # template_seq = [cut_image_resize(temp_img) for temp_img in template_seq]  # if 64*64
                    template_man_woman_seq.append(template_seq)
        template_seq_total = np.concatenate(template_man_woman_seq, 0)  # Total = 868 frames

    # ############################################################    
    def process_id(_id):
        id_dir = os.path.join(src_dir, _id)
        type_list = sorted(os.listdir(id_dir))
        for _type in type_list:
            type_dir = os.path.join(src_dir, _id, _type)
            view_list = sorted(os.listdir(type_dir))
            for _view in view_list:
                view_dir = os.path.join(src_dir, _id, _type, _view)
                des_view_dir = view_dir.replace(src_dir, des_dir)
                
                    
                sil_pkl = osp.join(view_dir, '{}.pkl'.format(_view))
                sil_seq = pickle.load(open(sil_pkl, 'rb'))
                num_sil = sil_seq.shape[0]
                
                # start_time = time.time()

                MinTempMatchDist_results, TempMatch_idx = template_match_dist_man_woman(sil_seq, template_seq_total)

                # end_time = time.time()
                # execution_time = end_time - start_time
                # print(f'Process each sequence for MaxConnect: {execution_time/num_sil}s')

                TempMatch_flag = True
                if len(TempMatch_idx) < 15:
                    TempMatch_idx = np.argsort(MinTempMatchDist_results)[:15]
                    TempMatch_flag = False

                new_sil_seq = sil_seq[TempMatch_idx, :, :]
                
                if not osp.exists(des_view_dir):
                    os.makedirs(des_view_dir)
                des_sil_pkl = osp.join(des_view_dir, '{}.pkl'.format(_view))
                pickle.dump(new_sil_seq, open(des_sil_pkl, 'wb'))

                #############################################################
                # for visualization analysis
                for i in range(num_sil):
                    if i not in TempMatch_idx:
                        sil = sil_seq[i]
                        drop_sil_name = 'drop_{}_{}_{}_dist_{:.2f}.png'.format(_id, _type, _view, MinTempMatchDist_results[i])
                        drop_sil_path = osp.join(drop_dir, drop_sil_name)
                        cv2.imwrite(drop_sil_path, sil)
                #############################################################
                # for qualitative analysis
                if len(TempMatch_idx) < num_sil:
                    print('ID={}, Type={}, View={}, Raw_Frame={}, Now_Frames={}, Drop_Frames={}, Drop_Ratio={:.2f}'.format( \
                            _id, _type, _view, num_sil, len(TempMatch_idx), num_sil-len(TempMatch_idx), 1-(len(TempMatch_idx)*1.0)/num_sil))
                if not TempMatch_flag:
                    print('TempMatch_idx={}, MinTempMatchDist_results={}'.format(TempMatch_idx, MinTempMatchDist_results[TempMatch_idx]))
                # if len(TempMatch_idx) == 0:
                #     print('Now_seq_frme=0, MinDist={}'.format(MinTempMatchDist_results))
                #     for i in range(num_sil):
                #         sil = sil_seq[i]
                #         drop_sil_name = 'drop_{}_{}_{}_dist_{:.2f}.png'.format(_id, _type, _view, MinTempMatchDist_results[i])
                #         drop_sil_path = osp.join(drop_dir, drop_sil_name)
                #         cv2.imwrite(drop_sil_path, sil)
                
                #############################################################
    
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
    print(f'Total time for Template Match: {execution_time}s')

