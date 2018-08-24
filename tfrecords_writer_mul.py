#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import numpy as np
import multiprocessing
import tensorflow as tf

from utils import listdir
from trans_sgf import sgf_list_kgs, sgf_list_cgos, trans_input

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class MyProcess(multiprocessing.Process):
    def __init__(self, process_id, name, lock, file_name):
        multiprocessing.Process.__init__(self)
        self.process_id = process_id
        self.name = name
        self.lock = lock
        self.file_name = file_name

    def run(self):
        print("开始进程：" + self.name)
        write_records(self.process_id, self.lock, self.file_name)
        print("退出进程：" + self.name)


def build_bytes_feature(v):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))


def build_int64_feature(v):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[v]))


def build_float_feature(v):
    return tf.train.Feature(int64_list=tf.train.FloatList(value=[v]))


def build_data(sample, label, legal):
    data = tf.train.Example(features=tf.train.Features(feature={
        'sample': build_bytes_feature(sample),
        'label': build_int64_feature(label),
        'legal': build_bytes_feature(legal)
    }))
    return data


def write_records(cnt, lock, file_name):
    LINE_NUM = 100000
    train_save_path = 'F:/go_data/training3'
    lock.acquire()
    if not os.path.exists(train_save_path):
        os.makedirs(train_save_path)
    lock.release()
    train_file_pattern = train_save_path + '/go_training_data_%.4d.tfrecords'
    train_file_no = cnt * 1000 + 1
    train_line_cnt = 0

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.per_process_gpu_memory_fraction = 0.05
    with tf.Session(config=config):
        train_writer = tf.python_io.TFRecordWriter(train_file_pattern % train_file_no)

        for f in file_name:
            with open(f, encoding='gb18030') as fp:
                s = fp.read()
                r, al = sgf_list_cgos(s)
                if r is not None and len(al) == 0:  # 不学让子棋
                    sample, label, legal, board = trans_input(r, al, cnt % 4)
                    if label is not None:
                        for i in range(len(label)):
                            train_writer.write(build_data(sample=np.array(sample[i], dtype=np.uint8).tobytes(),
                                                          label=label[i],
                                                          legal=np.array(legal[i], dtype=np.uint8).tobytes()).SerializeToString())
                            train_line_cnt += 1
                            if train_line_cnt >= LINE_NUM:  # 文件结束条件
                                train_writer.close()
                                train_line_cnt = 0
                                train_file_no += 1
                                train_writer = tf.python_io.TFRecordWriter(train_file_pattern % train_file_no)
        train_writer.close()


if __name__ == '__main__':
    file_src = 'F:/go_data/records3'
    file_name = []
    listdir(file_src, file_name)

    multi = 8
    # 创建新线程
    process_list = []
    lock = multiprocessing.Lock()
    for p in range(multi):
        t = MyProcess(p, "Process-" + str(p), lock, file_name[int(p / 4)::int(multi / 4)])
        t.start()
        process_list.append(t)

    for process in process_list:
        process.join()
