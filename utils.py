# -*- coding: utf-8 -*-
import os
import multiprocessing
from trans_sgf import sgf_list_cgos


def listdir(path, list_name):  # 传入存储的list
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file_path)


def read_data(files):
    for i in files:
        f = open(i)
        s = f.read()
        r, al = sgf_list_cgos(s)
    return 0


if __name__ == '__main__':
    file_src = 'F:/go_data/records3'
    file_name = []
    listdir(file_src, file_name)
    print(len(file_name))

    multi = 8
    # 创建新线程
    process_list = []
    for p in range(multi):
        t = multiprocessing.Process(target=read_data, args=(file_name[p::multi],))
        t.start()
        process_list.append(t)

    for process in process_list:
        process.join()
