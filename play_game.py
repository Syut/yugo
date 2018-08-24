#!/usr/bin/python
# -*- coding: utf-8 -*-
import random
import os
import numpy as np
import tensorflow as tf

from cnn_structure import conv_net
from trans_sgf import play_input, COO2SGF

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class PlayGame(object):
    def __init__(self, model_path, top_n=5, session_config=None):
        self._tf_session_config = session_config
        self.model_path = model_path
        self.saver = None
        self.top_n = top_n
        self.classes = 362

        self._init_graph()
        self._init_session()
        self._load_model()

    def _init_graph(self):
        # restore graph from meta
        self._tf_graph = tf.Graph()
        with self._tf_graph.as_default():
            x = tf.placeholder(tf.float32, [17, 19, 19], name='x_input')
            # Store layers weight & bias
            weights = {
                'wc1': tf.Variable(tf.random_normal([3, 3, 17, 92], stddev=0.05)),
                'wc2': tf.Variable(tf.random_normal([3, 3, 92, 384], stddev=0.05)),
                'wc3': tf.Variable(tf.random_normal([3, 3, 384, 512], stddev=0.05)),
                'wc4': tf.Variable(tf.random_normal([3, 1, 512, 512], stddev=0.05)),
                'wc5': tf.Variable(tf.random_normal([1, 3, 512, 512], stddev=0.05)),
                'wc6': tf.Variable(tf.random_normal([3, 1, 512, 512], stddev=0.05)),
                'wc7': tf.Variable(tf.random_normal([1, 3, 512, 512], stddev=0.05)),
                'wc8': tf.Variable(tf.random_normal([3, 1, 512, 512], stddev=0.05)),
                'wc9': tf.Variable(tf.random_normal([1, 3, 512, 512], stddev=0.05)),
                'wc10': tf.Variable(tf.random_normal([3, 1, 512, 512], stddev=0.05)),
                'wc11': tf.Variable(tf.random_normal([1, 3, 512, 512], stddev=0.05)),
                'wc12': tf.Variable(tf.random_normal([3, 1, 512, 512], stddev=0.05)),
                'wc13': tf.Variable(tf.random_normal([1, 3, 512, 512], stddev=0.05)),
                'wc14': tf.Variable(tf.random_normal([3, 1, 512, 512], stddev=0.05)),
                'wc15': tf.Variable(tf.random_normal([1, 3, 512, 512], stddev=0.05)),
                'wc16': tf.Variable(tf.random_normal([3, 1, 512, 512], stddev=0.05)),
                'wc17': tf.Variable(tf.random_normal([1, 3, 512, 512], stddev=0.05)),
                'wc18': tf.Variable(tf.random_normal([3, 1, 512, 512], stddev=0.05)),
                'wc19': tf.Variable(tf.random_normal([1, 3, 512, 512], stddev=0.05)),
                'wc20': tf.Variable(tf.random_normal([3, 1, 512, 512], stddev=0.05)),
                'wc21': tf.Variable(tf.random_normal([1, 3, 512, 512], stddev=0.05)),
                # fully connected
                'wd1': tf.Variable(tf.random_normal([512, 1024], stddev=0.04)),
                # 1024 inputs, 309 outputs (class prediction)
                'wout': tf.Variable(tf.random_normal([1024, self.classes], stddev=1 / 1024.0))
            }

            biases = {
                'bc1': tf.Variable(tf.random_normal([92])),
                'bc2': tf.Variable(tf.random_normal([384])),
                'bc3': tf.Variable(tf.random_normal([512])),
                'bc4': tf.Variable(tf.random_normal([512])),
                'bc5': tf.Variable(tf.random_normal([512])),
                'bc6': tf.Variable(tf.random_normal([512])),
                'bc7': tf.Variable(tf.random_normal([512])),
                'bc8': tf.Variable(tf.random_normal([512])),
                'bc9': tf.Variable(tf.random_normal([512])),
                'bc10': tf.Variable(tf.random_normal([512])),
                'bc11': tf.Variable(tf.random_normal([512])),
                'bc12': tf.Variable(tf.random_normal([512])),
                'bc13': tf.Variable(tf.random_normal([512])),
                'bc14': tf.Variable(tf.random_normal([512])),
                'bc15': tf.Variable(tf.random_normal([512])),
                'bc16': tf.Variable(tf.random_normal([512])),
                'bc17': tf.Variable(tf.random_normal([512])),
                'bc18': tf.Variable(tf.random_normal([512])),
                'bc19': tf.Variable(tf.random_normal([512])),
                'bc20': tf.Variable(tf.random_normal([512])),
                'bc21': tf.Variable(tf.random_normal([512])),
                'bd1': tf.Variable(tf.random_normal([1024])),
                'bout': tf.Variable(tf.random_normal([self.classes]))
            }

            restore_var = dict(weights, **biases)

            # Construct model
            pred = conv_net(x, weights, biases, 1., False)
            pred_top = tf.nn.top_k(tf.nn.softmax(pred), k=self.classes)
            tf.add_to_collection('pred_top', pred_top)

            sc = tf.get_collection("scale")
            bt = tf.get_collection("beta")
            pm = tf.get_collection("pop_mean")
            pv = tf.get_collection("pop_var")
            for i in range(len(sc)):
                restore_var['scale' + str(i)] = sc[i]
                restore_var['beta' + str(i)] = bt[i]
                restore_var['pop_mean' + str(i)] = pm[i]
                restore_var['pop_var' + str(i)] = pv[i]

            self.saver = tf.train.Saver(restore_var)

    def _init_session(self):
        # target_host = "//".join(("grpc:", "localhost:10935"))
        # self._tf_session = tf.Session(target_host, graph=self._tf_graph)
        if self._tf_session_config is None:
            config = tf.ConfigProto()
            config.allow_soft_placement = True
            config.gpu_options.per_process_gpu_memory_fraction = 0.6
            config.gpu_options.allow_growth = True
            self._tf_session_config = config

        self._tf_session = tf.Session(graph=self._tf_graph, config=self._tf_session_config)

    def _load_model(self):
        ckpt = tf.train.get_checkpoint_state(self.model_path)
        if ckpt is not None:
            self.saver.restore(self._tf_session, ckpt.model_checkpoint_path)
        else:
            print('Saver is None. Can\'t find model! path=', self.model_path)

    def get_one_hand(self, prcs, add_black, random_play=False):
        x_input, legal_label, color = play_input(prcs, add_black)
        pt = self._tf_session.run(self._tf_graph.get_collection('pred_top'),
                                  feed_dict={self._tf_graph.get_tensor_by_name("x_input:0"): x_input})
        legal_out, legal_prob = deal_legal(pt[0], legal_label)
        if len(legal_out) == 0:
            print("no legal hand!!!")
            return None, None
        out_label, prob = get_hand(legal_out, legal_prob, random_play)
        if out_label == 361:
            print('pass')
            out_hand = '%s[]' % color
        else:
            coo_x = int(out_label / 19)
            coo_y = out_label % 19
            # print(coo_x, coo_y)
            print(coo_x + 1, COO2SGF[coo_y])
            # print(COO2MGX[coo_y], 19 - coo_x)
            # with open(record_path2, 'a') as r:
            #     r.write('%s%d\n' % (COO2MGX[coo_y], 19 - coo_x))
            out_hand = '%s[%s%s]' % (color, COO2SGF[coo_x], COO2SGF[coo_y])
        return out_hand, prob

    def get_top_n_hand(self, prcs, add_black):
        x_input, legal_label, color = play_input(prcs, add_black)
        pt = self._tf_session.run(self._tf_graph.get_collection('pred_top'),
                                  feed_dict={self._tf_graph.get_tensor_by_name("x_input:0"): x_input})
        legal_out, legal_prob = deal_legal(pt[0], legal_label)
        if len(legal_out) == 0:
            print("no legal hand!!!")
            return None, None
        out_hands = []
        probs = []
        for i in range(min(len(legal_out), self.top_n)):
            if legal_out[i] == 361:
                print('pass')
                out_hand = '%s[]' % color
            else:
                coo_x = int(legal_out[i] / 19)
                coo_y = legal_out[i] % 19
                out_hand = '%s[%s%s]' % (color, COO2SGF[coo_x], COO2SGF[coo_y])
            out_hands.append(out_hand)
            probs.append(legal_prob[i])
        return out_hands, probs

    def close_session(self):
        self._tf_session.close()


def deal_legal(top_n_value, legal):
    legal_out = []
    legal_prob = []
    for i in range(len(legal)):
        if legal[top_n_value[1][0][i]] == 1:
            legal_out.append(top_n_value[1][0][i])
            legal_prob.append(top_n_value[0][0][i])
    legal_prob_norm = np.divide(legal_prob, np.sum(legal_prob))
    return legal_out, legal_prob_norm


def get_hand(legal_out, legal_prob, random_play):
    if not random_play:
        idx = np.where(legal_prob == np.max(legal_prob))[0][0]
        return legal_out[idx], legal_prob[idx]
    else:
        rd = random.random()
        out_hand = -1
        prob = 0
        for i in range(len(legal_out)):
            if rd < legal_prob[i] and out_hand < 0:
                out_hand = legal_out[i]
                prob = legal_prob[i]
                break
            else:
                rd -= legal_prob[i]
        return out_hand, prob


if __name__ == '__main__':
    # env = PlayGame('./play_model')
    env = PlayGame('./models/CNN20180730195145')
    record_path = './record.txt'
    with open(record_path, 'r') as r:
        prcs = r.read().lstrip(';')
        ab = ''
        hs, ps = env.get_one_hand(prcs, ab)

    """
    from trans_sgf import SGF2COO, COO2QQX, MGX2COO, COO2MGX, DEROLE, ROLE
    color = 1  # 对面的颜色
    
    # MultiGo
    go = ('a', 5)
    sgf_go = COO2SGF[19 - go[1]] + COO2SGF[MGX2COO[go[0]]]
    # sgf_go = ''
    # with open(record_path2, 'a') as r:
    #     r.write('%s%d\n' % (go[0], go[1]))
    with open(record_path, 'r') as r:
        before = r.read()
        if before == '' and color == -1:
            prcs = ''
        else:
            prcs = '%s%s[%s];' % (before, DEROLE[color], sgf_go)
    ab = ''
    hs, ps = env.get_one_hand(prcs, ab)
    # hs, ps = env.get_top_n_hand(prcs, ab)
    print(hs, ps)
    with open(record_path, 'a') as r:
        if prcs == '':
            r.write(hs + ';')
        else:
            r.write('%s[%s];%s;' % (DEROLE[color], sgf_go, hs))


    # QQ
    go = (16, 'p')
    sgf_go = COO2QQX[go[0]] + go[1]
    # sgf_go = ''
    with open(record_path, 'r') as r:
        before = r.read()
        if before == '' and color == -1:
            prcs = ''
        else:
            prcs = '%s%s[%s];' % (before, DEROLE[color], sgf_go)
    ab = ''
    hs, ps = env.get_one_hand(prcs, ab)
    # hs, ps = env.get_top_n_hand(prcs, ab)
    print(hs, ps)
    with open(record_path, 'a') as r:
        if prcs == '':
            r.write(hs + ';')
        else:
            r.write('%s[%s];%s;' % (DEROLE[color], sgf_go, hs))


    # 模仿棋flag
    flag = False
    # 以record为准
    for i in range(300):
        with open(record_path, 'r') as r:
            prcs = r.read().lstrip(';')
            ab = ''
            hs, ps = env.get_one_hand(prcs, ab)
            # hs, ps = env.get_top_n_hand(prcs, ab)
            print(hs, ps)
            with open(record_path, 'a') as r:
                r.write(';' + hs)
            # 模仿棋
            if flag:
                hs_trans = hs.split('[')[1]
                hs0 = COO2SGF[18 - SGF2COO[hs_trans[0]]]
                hs1 = COO2SGF[18 - SGF2COO[hs_trans[1]]]
                if hs0 == hs_trans[0] and hs1 == hs_trans[1]:
                    flag = False
                    continue
                else:
                    w_out = 'W[%s%s]' % (hs0, hs1)
                    print(w_out)
                    with open(record_path, 'a') as r:
                        r.write(';' + w_out)
    """
