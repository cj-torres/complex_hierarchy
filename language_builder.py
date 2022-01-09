import torch
from random import shuffle

abcd_map = {'a': 1, 'b': 2, 'c': 3, 'd': 4}

space = set(["a", "b"])

old = space
for n in range(20):
    new = set()
    for e in old:
        new.add(e + "a")
        new.add(e + "b")
    space = space.union(new)
    old = new


def dict_map(s):
    return abcd_map[s]


def anbn_checker(string):
    counter = 0
    switch = False
    for s in string:
        if s == "a":
            if switch:
                return False
            counter += 1
        if s == "b":
            counter -= 1
            switch = True
    if counter != 0:
        return False
    else:
        return True


def dyck_1_checker(string):
    counter = 0
    for s in string:
        if counter < 0:
            return False
        elif s == "a":
            counter += 1
        else:
            counter -= 1
    if counter != 0:
        return False
    else:
        return True


def anbm_checker(string):
    flip = False
    for s in string:
        if not flip:
            if s == "b":
                flip = True
        elif s == "a":
            return False
    return True


def abn_checker(string):
    if (len(string)%2) != 0:
        return False
    for i, s in enumerate(string):
        if (i%2==0) and s == "b":
            return False
        elif (i%2==1) and s == "a":
            return False
    return True


anbn = [(word, anbn_checker(word)) for word in space]
dyck1 = [(word, dyck_1_checker(word)) for word in space]
anbm = [(word, anbm_checker(word)) for word in space]
abn = [(word, abn_checker(word)) for word in space]


def make_io(lang):
    x = []
    y = []
    length = []
    for word in lang:
        x.append(torch.Tensor(list(map(dict_map, word[0]))))
        y.append(word[1])
        length.append(len(word[0])-1)
    x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True).type(torch.IntTensor)
    y = torch.FloatTensor(y)
    length = torch.LongTensor(length)

    return x, y, length


