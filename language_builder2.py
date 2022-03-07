import random
import torch


def dict_map(s):
    abcd_map = {'S': 1, 'a': 2, 'b': 3, 'c': 4, 'd': 5}
    return abcd_map[s]


def geometric(p, counts=0):
    if random.uniform(0,1)>p:
        return geometric(p, counts+1)
    else:
        return counts

#checkers
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


#Dyck-1
def dyck1_transition(s, p=.5, q=.25):
    assert(p+q < 1)
    new_s = ""
    for c in s:
        if c == "S":
            n = random.uniform(0,1)
            new_c = ""+(0 <= n < p)*"aSb"+(p <= n < p+q)*"SS"
        else:
            new_c = c
        new_s += new_c
    return new_s


def gen_dyck1(max_length):
    a = "S"
    while "S" in a:
        a = dyck1_transition(a)
        if len(a.replace("S","")) >= max_length:
            a = "S"
    return "S"+a


def gen_dyck1_words(N, max_length):
    dyck1 = set()
    while len(dyck1) < N+1:
        dyck1.add(gen_dyck1(max_length))
        print(len(dyck1))
    dyck1.remove("S")
    return dyck1


def gen_dyck1_words_redundant(N, max_length):
    dyck1 = []
    while len(dyck1) < N+1:
        dyck1.append(gen_dyck1(max_length))
        #print(len(dyck1))
    #dyck1.remove("S")
    return dyck1


def make_dyck1_continuation(w):
    cont = []
    for i, c in enumerate(w):
        a_count =  len(w[:i+1].replace("b",""))-1
        b_count = len(w[:i+1].replace("a",""))-1
        if a_count == b_count:
            cont.append([1, 1, 0])
        elif a_count > b_count:
            cont.append([0, 1, 1])
    return cont


def gen_rand_words_not_dyck1(N):
    rand_words = set()
    while len(rand_words) < N:
        n = random.randint(2,50)
        w = ""
        for i in range(n):
            x = random.randint(0,1)
            w = w + "a"*x + "b"*(1-x)
        if not dyck_1_checker(w):
            rand_words.add(w)
        #print(len(rand_words))
    return rand_words


def make_dyck1_io(N):
    dyck1 = gen_dyck1_words(N)
    not_dyck1 = gen_rand_words_not_dyck1(N)
    x = []
    y = []
    length = []
    for word in dyck1:
        x.append(torch.Tensor(list(map(dict_map, word))))
        y.append(1)
        length.append(len(word) - 1)
    for word in not_dyck1:
        x.append(torch.Tensor(list(map(dict_map, word))))
        y.append(0)
        length.append(len(word) - 1)
    x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True).type(torch.IntTensor)
    y = torch.FloatTensor(y)
    length = torch.LongTensor(length)

    return x, y, length


def make_dyck1_io_cont(N):
    dyck1 = gen_dyck1_words(N)
    x = []
    y = []
    mask = []
    for word in dyck1:
        x.append(torch.Tensor(list(map(dict_map, word))))
        y.append(torch.Tensor(make_dyck1_continuation(word)))
        mask.append(torch.ones(len(word)))
    x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True).type(torch.IntTensor)
    y = torch.nn.utils.rnn.pad_sequence(y, batch_first=True).type(torch.IntTensor)
    mask = torch.nn.utils.rnn.pad_sequence(mask, batch_first=True).type(torch.BoolTensor)

    return x, y, mask


def make_dyck1_io_cont_redundant(N, max_length):
    dyck1 = gen_dyck1_words_redundant(N, max_length)
    x = []
    y = []
    mask = []
    for word in dyck1:
        x.append(torch.Tensor(list(map(dict_map, word))))
        y.append(torch.Tensor(make_dyck1_continuation(word)))
        mask.append(torch.ones(len(word)))
    x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True).type(torch.IntTensor)
    y = torch.nn.utils.rnn.pad_sequence(y, batch_first=True).type(torch.IntTensor)
    mask = torch.nn.utils.rnn.pad_sequence(mask, batch_first=True).type(torch.BoolTensor)

    return x, y, mask


#anbn
def gen_anbn_words(N):
    anbn = set()
    for i in range(N):
        anbn.add("S"+"a"*(i+1) + "b"*(i+1))
    return anbn


def gen_anbn_words_redundant(N, p):
    anbn = []
    for i in range(N):
        n = geometric(p)
        anbn.append("S"+"a"*n+"b"*n)
    return anbn


def gen_rand_words_not_anbn(N):
    rand_words = set()
    while len(rand_words) < N:
        n = random.randint(2, 2*N)
        w = ""
        for i in range(n):
            x = random.randint(0,1)
            w = w + "a"*x + "b"*(1-x)
        if not anbn_checker(w):
            rand_words.add(w)
        #print(len(rand_words))
    return rand_words


def make_anbn_continuation(w):
    length = len(w) - 1
    cont = [[0,1,1]] * int(length/2) + [[0,0,1]] * int(length/2)
    cont.insert(0,[1, 1, 0])
    cont[-1] = [1,0,0]
    return cont


def make_anbn_io(N):
    anbn = gen_anbn_words(N)
    not_anbn = gen_rand_words_not_anbn(N)
    x = []
    y = []
    length = []
    for word in anbn:
        x.append(torch.Tensor(list(map(dict_map, word))))
        y.append(1)
        length.append(len(word) - 1)
    for word in not_anbn:
        x.append(torch.Tensor(list(map(dict_map, word))))
        y.append(0)
        length.append(len(word) - 1)
    x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True).type(torch.IntTensor)
    y = torch.FloatTensor(y)
    length = torch.LongTensor(length)

    return x, y, length


def make_anbn_io_cont(N):
    anbn = gen_anbn_words(N)
    x = []
    y = []
    mask = []
    for word in anbn:
        x.append(torch.Tensor(list(map(dict_map, word))))
        y.append(torch.Tensor(make_anbn_continuation(word)))
        mask.append(torch.ones(len(word)))
    x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True).type(torch.IntTensor)
    y = torch.nn.utils.rnn.pad_sequence(y, batch_first=True).type(torch.IntTensor)
    mask = torch.nn.utils.rnn.pad_sequence(mask, batch_first=True).type(torch.BoolTensor)

    return x, y, mask


def make_anbn_io_cont_redundant(N, p):
    anbn = gen_anbn_words_redundant(N, p)
    x = []
    y = []
    mask = []
    for word in anbn:
        x.append(torch.Tensor(list(map(dict_map, word))))
        y.append(torch.Tensor(make_anbn_continuation(word)))
        mask.append(torch.ones(len(word)))
    x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True).type(torch.IntTensor)
    y = torch.nn.utils.rnn.pad_sequence(y, batch_first=True).type(torch.IntTensor)
    mask = torch.nn.utils.rnn.pad_sequence(mask, batch_first=True).type(torch.BoolTensor)

    return x, y, mask


# anbm
def gen_anbm_words_redundant(N, p):
    anbm = []
    for i in range(N):
        n = geometric(p)
        m = geometric(p)
        anbm.append("S"+"a"*n+"b"*m)
    return anbm


def make_anbm_continuation(w):
    length = len(w)-1
    n = len(w.replace("b", ""))-1
    m = length - n
    cont = [[1, 1, 1]] * n + [[1, 0, 1]] * m
    return cont


def make_anbm_io_cont_redundant(N, p):
    anbm = gen_anbm_words_redundant(N, p)
    x = []
    y = []
    mask = []
    for word in anbm:
        x.append(torch.Tensor(list(map(dict_map, word))))
        y.append(torch.Tensor(make_anbm_continuation(word)))
        mask.append(torch.ones(len(word)))
    x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True).type(torch.IntTensor)
    y = torch.nn.utils.rnn.pad_sequence(y, batch_first=True).type(torch.IntTensor)
    mask = torch.nn.utils.rnn.pad_sequence(mask, batch_first=True).type(torch.BoolTensor)

    return x, y, mask


#(ab)n
def gen_abn_words_redundant(N, p):
    abn = []
    for i in range(N):
        n = geometric(p)
        abn.append("S"+"ab"*n)
    return abn


def make_abn_continuation(w):
    n = (len(w)-1)/2
    cont = [[1, 1, 0], [1, 0, 1]] * n + [[1, 1, 0]]
    return cont


def make_abn_io_cont_redundant(N, p):
    abn = gen_abn_words_redundant(N, p)
    x = []
    y = []
    mask = []
    for word in abn:
        x.append(torch.Tensor(list(map(dict_map, word))))
        y.append(torch.Tensor(make_abn_continuation(word)))
        mask.append(torch.ones(len(word)))
    x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True).type(torch.IntTensor)
    y = torch.nn.utils.rnn.pad_sequence(y, batch_first=True).type(torch.IntTensor)
    mask = torch.nn.utils.rnn.pad_sequence(mask, batch_first=True).type(torch.BoolTensor)

    return x, y, mask


#Dyck-2
def dyck2_transition(s):
    new_s = ""
    for c in s:
        if c == "S":
            n = random.uniform(0,1)
            new_c = ""+(0 <= n < .25)*"aSb"+(.25 <= n < .5)*"cSd"+(.5 <= n < .75)*"SS"
        else:
            new_c = c
        new_s += new_c
    return new_s


def gen_dyck2():
    a = "S"
    while "S" in a:
        a = dyck2_transition(a)
        if len(a.replace("S","")) >= 50:
            a = "S"
    return "S"+a


def gen_dyck2_words(N):
    dyck2 = set()
    while len(dyck2) < N:
        dyck2.add(gen_dyck2())
        #print(len(dyck2))
    return dyck2