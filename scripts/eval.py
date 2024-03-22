import csv
import sys, os
import numpy as np
import math
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '..' ))
sys.path.append(BASE_DIR)
from utils.pcdpy3 import load_pcd
from utils import cnt_staticAdynamic

DATA_FOLDER = "/home/mjiaab/workspace/edo_ws/edomap_release/edomap/data/KITTI"
METHODS_NAME = "beautymap"
SEQUENCE_SELECT = ["05"]
# DATA_FOLDER = "/home/mjiaab/workspace/edo_ws/edomap_release/edomap/data/cones_two_people"
# METHODS_NAME = "beautymap"
# SEQUENCE_SELECT = [""]

# Step 1: Read your existing csv file
with open(f'{BASE_DIR}/scripts/benchmark_results_HA.csv', 'r') as file:
    reader = csv.DictReader(file)
    data = list(reader)

# Step 2: Add a new method to the data
for sequence in SEQUENCE_SELECT:
    print("Processing: ", sequence, " with method: ", METHODS_NAME)
    gt_pcd_path = f"{DATA_FOLDER}/{sequence}/gt_cloud.pcd"
    et_pcd_path = f"{DATA_FOLDER}/{sequence}/eval/{METHODS_NAME}_output_exportGT.pcd"
    assert os.path.exists(et_pcd_path), f"{et_pcd_path} does not exist. Please run c++ `export_eval_pcd`"
    row_dict = {'Sequence': sequence, 'Methods': METHODS_NAME}
    gt_pc_ = load_pcd(gt_pcd_path)
    num_gt = cnt_staticAdynamic(gt_pc_.np_data)
    et_pc_ = load_pcd(et_pcd_path)

    assert et_pc_.np_data.shape[0] == gt_pc_.np_data.shape[0] , \
        f"{et_pcd_path} Error: The number of points in et_pc_ and gt_pc_ do not match.\
        \nThey must match for evaluation, if not Please run `export_eval_pcd`."
    
    right_dynamic = np.count_nonzero((et_pc_.np_data[:,3] == 1) * (gt_pc_.np_data[:,3] == 1)) # TP
    wrong_dynamic = np.count_nonzero((et_pc_.np_data[:,3] == 1) * (gt_pc_.np_data[:,3] == 0)) # FP
    right_static = np.count_nonzero((et_pc_.np_data[:,3] == 0) * (gt_pc_.np_data[:,3] == 0)) # TN
    wrong_static = np.count_nonzero((et_pc_.np_data[:,3] == 0) * (gt_pc_.np_data[:,3] == 1)) # FN

    static_accuracy = float(right_static) / float(num_gt['static']) * 100
    dynamic_accuracy = float(right_dynamic) / float(num_gt['dynamic']) * 100
    # AA = math.sqrt(static_accuracy*dynamic_accuracy)
    HA = 2 * static_accuracy*dynamic_accuracy/ (static_accuracy + dynamic_accuracy)


    row_dict['# TN'], row_dict['# FN'], row_dict['# TP'], row_dict['# FP'], row_dict['SA ↑'], row_dict['DA ↑'], row_dict['HA ↑'] =\
        right_static, wrong_static, right_dynamic, wrong_dynamic, static_accuracy, dynamic_accuracy, HA
    data.append(row_dict)

import csv
from tabulate import tabulate
# print
sequences = set([row['Sequence'] for row in data])
for sequence in SEQUENCE_SELECT: #sequences:
    filtered_data = [row for row in data if row['Sequence'] == sequence]
    table_data = [[row['Methods'], row['# TN'], row['# TP'], row['SA ↑'], row['DA ↑'], row['HA ↑']] for row in filtered_data]

    # print the data
    print('Sequence: ', sequence)
    print(tabulate(table_data, headers=['Methods', '# TN', '# TP', 'SA ↑', 'DA ↑', 'HA ↑'], tablefmt='orgtbl'))
    print('='*20, ' Friendly dividing line ^v^ ', '='*20, '\n')
