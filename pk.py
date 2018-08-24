from play_game import PlayGame

# black = PlayGame('./models/CNN20180730195145')
# white = PlayGame('./play_model')

black = PlayGame('./play_model')
white = PlayGame('./models/CNN20180730195145')
record_path = './record.txt'

for i in range(200):
    with open(record_path, 'r') as r:
        prcs = r.read().lstrip(';')
        ab = ''
        hs, ps = black.get_one_hand(prcs, ab)
        # hs, ps = env.get_top_n_hand(prcs, ab)
        print(hs, ps)
        with open(record_path, 'a') as r:
            r.write(';' + hs)

    with open(record_path, 'r') as r:
        prcs = r.read().lstrip(';')
        ab = ''
        hs, ps = white.get_one_hand(prcs, ab)
        # hs, ps = env.get_top_n_hand(prcs, ab)
        print(hs, ps)
        with open(record_path, 'a') as r:
            r.write(';' + hs)
