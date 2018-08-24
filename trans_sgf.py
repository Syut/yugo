import numpy as np
from kill import kill

SGF2COO = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9,
           'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15, 'q': 16, 'r': 17, 's': 18}
COO2SGF = dict(zip(SGF2COO.values(), SGF2COO.keys()))

ROLE = {'B': 1, 'W': -1}
DEROLE = dict(zip(ROLE.values(), ROLE.keys()))

QQX2COO = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10,
           'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15, 'p': 16, 'q': 17, 'r': 18, 's': 19}
COO2QQX = dict(zip(QQX2COO.values(), QQX2COO.keys()))

MGX2COO = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'j': 8, 'k': 9,
           'l': 10, 'm': 11, 'n': 12, 'o': 13, 'p': 14, 'q': 15, 'r': 16, 's': 17, 't': 18}
COO2MGX = dict(zip(MGX2COO.values(), MGX2COO.keys()))


def sgf_list_cgos(sgf):
    ret = []
    record = sgf.rstrip('\n').rstrip(')').split(";")
    info = record[1]
    add_black = info.split("AB")
    if len(add_black) == 2:
        add_list = add_black[-1].replace('\n', '').lstrip('[').rstrip(']').split('][')
    else:
        add_list = []
    if record[-1].find('C') > -1:
        rec = record[2:-1]
    else:
        rec = record[2:]
    for hand in rec:
        h, t = hand.split('L')
        if len(h) > 6:
            print('error!!!!%s' % sgf)
            return None, None
        else:
            ret.append(h[:-1])
    return ret, add_list


def sgf_list_kgs(sgf):
    record = sgf.rstrip('\n').rstrip(')').split(";")
    info = record[1]
    add_black = info.split("AB")
    if len(add_black) == 2:
        add_list = add_black[-1].replace('\n', '').lstrip('[').rstrip(']').split('][')
    else:
        add_list = []
    if record[-1].find('TW') > -1 or record[-1].find('TB') > -1:
        return record[2:-1], add_list
    else:
        if len(record[-1]) > 5:
            print('-------------')
            print(sgf)
            return None, None
        else:
            return record[2:], add_list


# 旋转坐标
def transpose(x, y, num):
    if num == 0:
        return x, y
    elif num == 1:
        return y, 18 - x
    elif num == 2:
        return 18 - x, 18 - y
    elif num == 3:
        return 18 - y, x
    else:
        print('num is in [0,1,2,3] represent 0 -90 -180 -270')
        return None


def trans_input(record, add_black, trans_num=0):
    # return
    batch_sample = []
    batch_label = []
    batch_legal = []

    board = np.zeros((19, 19), dtype=np.int32)
    for ab in add_black:
        ab_x, ab_y = transpose(SGF2COO[ab[0]], SGF2COO[ab[1]], trans_num)
        board[ab_x][ab_y] = 1

    sample_tower = [np.copy(board)]
    color = -1 if len(add_black) > 0 else 1
    sample, legal = one_sample(sample_tower, board, color, -2)
    batch_sample.append(sample)
    batch_legal.append(legal)

    for h in record:
        c, hand = h.strip(']').split('[')
        color = ROLE[c]
        if hand == '':
            if len(sample_tower) > 2:
                sample_tower.append(sample_tower[-2])
            else:
                print('=================')
                print(record)
                return None, None, None, None
            coo_x = 19
            coo_y = 0
        else:
            if SGF2COO.get(hand[0]) is None:
                print('=================')
                print(record)
                return None, None, None, None
            if SGF2COO.get(hand[1]) is None:
                print('=================')
                print(record)
                return None, None, None, None
            coo_x, coo_y = transpose(SGF2COO[hand[0]], SGF2COO[hand[1]], trans_num)
            board[coo_x][coo_y] = color
            kill(-color, board)

            one_color_board = np.zeros_like(board)
            m = board == color
            one_color_board[m] = 1
            sample_tower.append(one_color_board)

        last_hand = batch_label[-3] if len(batch_sample) > 3 else -2
        next_color = -color
        sample, legal = one_sample(sample_tower, board, next_color, last_hand)
        batch_sample.append(sample)
        batch_label.append(coo_x * 19 + coo_y)
        batch_legal.append(legal)

    return batch_sample, batch_label, batch_legal, board


def one_sample(sample_tower, board, color, last_hand):
    if len(sample_tower) < 16:
        sample = sample_tower.copy()
        sample.reverse()
        while len(sample) < 16:
            sample.append(sample_tower[0])
    else:
        sample = sample_tower[-16:]
        sample.reverse()
    # sample
    is_black_layer = np.ones_like(board) if color == 1 else np.zeros_like(board)
    sample.append(is_black_layer)
    # legal
    legal = legal_label(color, board, last_hand)
    return sample, legal


def legal_label(color, board, last_hand):
    """
    判断合法手1合法，0不合法
    1.该位置为空
    2.点下去自己会被提的位置，不合法
    3.不是刚打劫的位置
    :param color:
    :param board:
    :param last_hand:
    :return:
    """
    legal = np.zeros(362, dtype=np.int32)
    legal[361] = 1  # pass永远合法
    for i in range(19):
        for j in range(19):
            if board[i][j] == 0 and last_hand != i * 19 + j:
                # 没有走在眼里
                if i > 0 and board[i - 1][j] == 0:
                    legal[i * 19 + j] = 1
                    continue
                if i + 1 < 19 and board[i + 1][j] == 0:
                    legal[i * 19 + j] = 1
                    continue
                if j > 0 and board[i][j - 1] == 0:
                    legal[i * 19 + j] = 1
                    continue
                if j + 1 < 19 and board[i][j + 1] == 0:
                    legal[i * 19 + j] = 1
                    continue
                # 这是个“眼”
                check_board = np.copy(board)
                check_board[i][j] = color
                # 检查能否提对面的棋子
                kill(color, check_board)
                if i > 0 and board[i - 1][j] == 0:
                    legal[i * 19 + j] = 1
                    continue
                if i + 1 < 19 and board[i + 1][j] == 0:
                    legal[i * 19 + j] = 1
                    continue
                if j > 0 and board[i][j - 1] == 0:
                    legal[i * 19 + j] = 1
                    continue
                if j + 1 < 19 and board[i][j + 1] == 0:
                    legal[i * 19 + j] = 1
                    continue
                # 检查自己是否会被提
                kill(-color, check_board)
                if check_board[i][j] == color:
                    legal[i * 19 + j] = 1
    return legal


def play_input(prcs, add_black):
    prcs = prcs.rstrip(';')
    add_black = add_black.rstrip(';')
    prcs_list = [] if prcs == '' else prcs.split(';')
    ab_list = [] if add_black == '' else add_black.split(';')

    init_board = np.zeros((19, 19), dtype=np.int32)
    if len(ab_list) > 0:  # 这里的color指上一步的color!!!
        color = 1 if len(prcs_list) == 0 else -1
        for ab in ab_list:
            init_board[SGF2COO[ab[0]]][SGF2COO[ab[1]]] = 1
    else:
        color = -1

    board = np.copy(init_board)

    sample = []
    eat = []
    if len(prcs_list) < 16:
        for p in prcs_list:
            c, hand = p.strip(']').split('[')
            color = ROLE[c]
            if hand != '':
                coo_x = SGF2COO[hand[0]]
                coo_y = SGF2COO[hand[1]]
                board[coo_x][coo_y] = color
                eat = kill(color, board)

                one_color_board = np.zeros_like(board)
                m = board == color
                one_color_board[m] = 1
                sample.append(one_color_board)
            else:
                if len(sample) > 1:
                    sample.append(sample[-2])
                else:
                    one_color_board = np.zeros_like(board)
                    m = board == color
                    one_color_board[m] = 1
                    sample.append(one_color_board)
        sample.reverse()
        while len(sample) < 16:
            sample.append(init_board)
    else:
        for p in prcs_list[:-16]:
            c, hand = p.strip(']').split('[')
            color = ROLE[c]
            if hand != '':
                board[SGF2COO[hand[0]]][SGF2COO[hand[1]]] = color
                eat = kill(color, board)
        for p in prcs_list[-16:]:
            c, hand = p.strip(']').split('[')
            color = ROLE[c]
            if hand != '':
                coo_x = SGF2COO[hand[0]]
                coo_y = SGF2COO[hand[1]]
                board[coo_x][coo_y] = color
                eat = kill(color, board)

                one_color_board = np.zeros_like(board)
                m = board == color
                one_color_board[m] = 1
                sample.append(one_color_board)
            else:
                if len(sample) > 1:
                    sample.append(sample[-2])
                else:
                    one_color_board = np.zeros_like(board)
                    m = board == color
                    one_color_board[m] = 1
                    sample.append(one_color_board)
        sample.reverse()

    next_color = -color
    is_black_layer = np.ones_like(board) if next_color == 1 else np.zeros_like(board)
    sample.append(is_black_layer)

    last_hand = eat[0] if len(eat) == 1 else -1  # 打劫

    legal = legal_label(next_color, board, last_hand)
    # see_legal = legal[:-1].reshape((19, 19))
    return sample, legal, DEROLE[next_color]


if __name__ == "__main__":
    data_path = 'E:/go_data/records/kgs-19-2018-05-new'
    # 1,64,73
    f = open(data_path + "/2018-05-01-64.sgf")
    s = f.read()
    r, al = sgf_list_kgs(s)
    if len(al) == 0:
        b = trans_input(r, al)
        print(b)

        # prcs = 'B[qd];W[qf];B[qh]'
        # add_black = ''
        # s, l, c = play_input(prcs, add_black)
