
'''
# Created: 2023-1-15 12:14
# Copyright (C) 2022-now, RPL, KTH Royal Institute of Technology
# Author: Kin ZHANG  (https://kin-zhang.github.io/)

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

Python Timer from https://stackoverflow.com/a/26695514/9281669
Modified a little by Kin
'''
import time

def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def TOC(chat = "Default", tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "\033[1m\x1b[34m[%-15.15s] takes %10f ms\033[0m" %(chat, tempTimeInterval*1000))
        TIC()

def TIC():
    # Records a time in TicToc, marks the beginning of a time interval
    TOC(tempBool=False)

import os
def mkdir_folder(path, sensor_type):
    for s_type in sensor_type:
        if not os.path.isdir(os.path.join(path, s_type)):
            os.makedirs(os.path.join(path, s_type))
    return True

class bc:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# reference: https://stackoverflow.com/questions/26079881/kl-divergence-of-two-gmms?noredirect=1&lq=1
# TODO sample again not smart way? is there any continues way to do that?
def gmm_kl(gmm_p, gmm_q, n_samples=10**3):
    X = gmm_p.sample(n_samples)
    log_p_X = gmm_p.score_samples(X[0])
    log_q_X = gmm_q.score_samples(X[0])
    return log_p_X.mean() - log_q_X.mean()

import operator

def SELECT_Ptsindex_from_matrix(all3d_indexs, threeD2ptindex, i, j, min_i_map, min_j_map):
    if (len(all3d_indexs)==0):
        return []
    elif(len(all3d_indexs)==1):
        return threeD2ptindex[i+min_i_map][j+min_j_map][all3d_indexs[0]]
    else:
        tupleOfTuples = operator.itemgetter(*all3d_indexs)(threeD2ptindex[i+min_i_map][j+min_j_map])
        return [element for tupl in tupleOfTuples for element in tupl]
