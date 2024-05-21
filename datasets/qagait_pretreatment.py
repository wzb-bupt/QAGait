# import wild_connect_area
# import wild_minAreaRect
# import wild_template

import time
import os
import subprocess
import sys
from datetime import datetime
import argparse
import os.path as osp
import logging

'''
Usages:
    ##############
    ### Gait3D ###
    ##############
        python qagait_pretreatment.py \
            -i /data/wangzengbin/Gait3D/Silhouette/2D_Silhouettes \
            -o /data/wangzengbin/Gait3D/Silhouette/QA-Gait3D-Release-20240521 \
            -template /data/wangzengbin/Template/Template-frames-30-height-0-1-2-pkl/ \
            -d Gait3D

    ##############
    ###  GREW  ###
    ##############
        python qagait_pretreatment.py \
            -i /data/wangzengbin/GREW/GREW-Sihouette-Rearange \
            -o /data/wangzengbin/GREW/QA-GREW-Release-20240521 \
            -template /data/wangzengbin/Template/Template-frames-30-height-0-1-2-pkl/ \
            -d GREW

    ##############
    ## CASIA-B  ##
        python qagait_pretreatment.py \
            -i /data/wangzengbin/CASIA-B/Silhouette/GaitDatasetB-silh \
            -o /data/wangzengbin/CASIA-B/Silhouette/QA-CASIA-B-Release-20240521 \
            -template /data/wangzengbin/Template/Template-frames-30-height-0-1-2-pkl/ \
            -d CASIA-B 
'''

class FlushFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

def setup_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Create a file handler and set its level to INFO
    fh = FlushFileHandler(log_file, mode='w')
    fh.setLevel(logging.INFO)
    
    # Create a console handler and set its level to INFO
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    
    # Create a formatter
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # Add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def run_script(command, stage, logger):
    logger.info(f"=========== {stage} ===========")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in iter(process.stdout.readline, b''):
        logger.info(line.decode('utf-8').strip())
    process.stdout.close()
    process.wait()
    
    if process.returncode != 0:
        logger.error(f"Script {' '.join(command)} failed with return code {process.returncode}")
        sys.exit(process.returncode)
    else:
        logger.info(f"Script {' '.join(command)} completed successfully")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Connection Area for Wild Gait Dataset')
    parser.add_argument('-i', '--input_path', default='', type=str, help='Input Path')
    parser.add_argument('-o', '--output_root', default='', type=str, help='Output Root')
    parser.add_argument('-template', '--template_path', default='/data/wangzengbin/Template/Template-frames-30-height-0-1-2-pkl/', type=str, help='Template Path')
    parser.add_argument('-d', '--dataset_name', default='casiab', type=str, help='Dataset Name')
    # parser.add_argument('-l', '--log_file', default='', type=str, help='Log file path. Default: ./pretreatment.log')
    # parser.add_argument('--MaxConnect_drop_path', default='', type=str, help='MaxConnect Drop Path')
    # parser.add_argument('--TemplateMatch_drop_path', default='', type=str, help='Template Match Drop Path')
    # parser.add_argument('--Align_drop_path', default='', type=str, help='Alignment Drop Path')
    args = parser.parse_args()

    # parser args
    INPUT_PATH = args.input_path
    OUTPUT_ROOT = args.output_root
    Template_Path = args.template_path
    Dataset_Name = args.dataset_name

    # make output directory
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # Init log file
    current_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    log_file = f'QAGait-Pretreatment-{Dataset_Name}-{current_time}.log'
    logger = setup_logger(log_file)

    # Stage-1: MaxConnect (Maximal Connected Area)
    stage_1_command = [
        'python', '-u', 'wild_connect_area.py',
        '--input_path', INPUT_PATH,
        '--output_path', osp.join(OUTPUT_ROOT, Dataset_Name+'-MaxConnect'),
        '--drop_path', osp.join(OUTPUT_ROOT, Dataset_Name+'-MaxConnect-Drop')
    ]
    run_script(stage_1_command, "Stage-1: wild_connect_area.py", logger)

    # Stage-2: pretreatment.py
    stage_2_command = [
        'python', 'pretreatment.py',
        '--input_path', osp.join(OUTPUT_ROOT, Dataset_Name+'-MaxConnect'),
        '--output_path', osp.join(OUTPUT_ROOT, Dataset_Name+'-MaxConnect-pkl')
    ]
    run_script(stage_2_command, "Stage-2: pretreatment.py", logger)  

    # Stage-3: Template Match
    stage_3_command = [
        'python', '-u', 'wild_template.py',
        '--input_path', osp.join(OUTPUT_ROOT, Dataset_Name+'-MaxConnect-pkl'),
        '--output_path', osp.join(OUTPUT_ROOT, Dataset_Name+'-MaxConnect-TemporalMatch-pkl'),
        '--template_path', Template_Path,
        '--drop_path', osp.join(OUTPUT_ROOT, Dataset_Name+'-TemporalMatch-Drop'),
        '--threshold', '100',
        '--height_num', '3'
    ]
    run_script(stage_3_command, "Stage-3: wild_template.py", logger)

    # Stage-4: Align
    stage_4_command = [
        'python', '-u', 'wild_minAreaRect.py',
        '--input_path', osp.join(OUTPUT_ROOT, Dataset_Name+'-MaxConnect-TemporalMatch-pkl'),
        '--output_path', osp.join(OUTPUT_ROOT, Dataset_Name+'-MaxConnect-TemporalMatch-Align-pkl'),
        '--align_results_path', osp.join(OUTPUT_ROOT, Dataset_Name+'-Large-Angle'),
        '--align_type', 'sequence-level', 
        '--disturb', '0',
        '--seq_restrict_flg'
    ]
    run_script(stage_4_command, "Stage-4: wild_minAreaRect.py", logger)