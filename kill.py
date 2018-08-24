def kill(color, board):
    """
    走子后计算提子
    :param color:走子的颜色 black:1,white:-1
    :param board: 
    :return: eat:吃掉的子
    """
    search_position = []
    eat = []
    for i in range(19):
        for j in range(19):
            if board[i][j] != -color or (board[i][j] == -color and i * 19 + j in search_position):
                continue
            else:
                block = []
                block.append(i * 19 + j)
                expand(i, j, board, block)
                search_position.extend(block)
                if has_qi(board, block):
                    continue
                else:
                    for b in block:
                        board[int(b / 19)][b % 19] = 0
                        eat.append(b)
    return eat


def expand(i, j, board, block):
    # Left
    if i - 1 >= 0 and board[i - 1][j] == board[i][j] and ((i - 1) * 19 + j not in block):
        block.append((i - 1) * 19 + j)
        expand(i - 1, j, board, block)
    # Up
    if j - 1 >= 0 and board[i][j - 1] == board[i][j] and (i * 19 + j - 1 not in block):
        block.append(i * 19 + j - 1)
        expand(i, j - 1, board, block)
    # Right
    if i + 1 < 19 and board[i + 1][j] == board[i][j] and ((i + 1) * 19 + j not in block):
        block.append((i + 1) * 19 + j)
        expand(i + 1, j, board, block)
    # Down
    if j + 1 < 19 and board[i][j + 1] == board[i][j] and (i * 19 + j + 1 not in block):
        block.append(i * 19 + j + 1)
        expand(i, j + 1, board, block)


def has_qi(board, block):
    for b in block:
        i = int(b / 19)
        j = b % 19
        if i - 1 >= 0 and board[i - 1][j] == 0:
            return True
        if i + 1 < 19 and board[i + 1][j] == 0:
            return True
        if j - 1 >= 0 and board[i][j - 1] == 0:
            return True
        if j + 1 < 19 and board[i][j + 1] == 0:
            return True
    return False


if __name__ == "__main__":
    import numpy as np
    from trans_sgf import legal_label

    # new_board = np.zeros((19, 19), dtype=np.int32)
    new_board = [[0, 1, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [1, 0, 1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 1, -1, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    color = -1  # 走子的颜色 black:1,white:-1
    kill(color, new_board)
    print(np.array(new_board))
    print(legal_label(color, new_board, -1))
