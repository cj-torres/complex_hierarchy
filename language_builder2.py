import random
import torch
from collections import Counter


def dict_map(s):
    abcd_map = {'S': 1, 'a': 2, 'b': 3, 'c': 4, 'd': 5}
    return abcd_map[s]


def geometric(p, counts=0):
    # recursive function to return samples from a geometric distribution
    if random.uniform(0, 1) > p:
        return geometric(p, counts+1)
    else:
        return counts


def shuffler(language):
    N = len(language)
    lang_counts = Counter(language)
    shuffled = list(lang_counts.keys())
    random.shuffle(shuffled)
    shuffled_lang = []
    for key in shuffled:
        shuffled_lang += [key]*lang_counts[key]
    assert (len(shuffled_lang) == N)
    return shuffled_lang


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


def gen_dyck1_words_redundant(N, max_length):
    dyck1 = []
    while len(dyck1) < N:
        dyck1.append(gen_dyck1(max_length))
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


def make_dyck1_io_cont_redundant(N, max_length):
    dyck1 = gen_dyck1_words_redundant(N, max_length)
    dyck1.sort(key=len)
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


def make_dyck1_io_cont_shuffled(N, max_length):
    dyck1 = gen_dyck1_words_redundant(N, max_length)
    dyck1 = shuffler(dyck1)
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
def gen_anbn_words_redundant(N, p):
    anbn = []
    for i in range(N):
        n = geometric(p)
        anbn.append("S"+"a"*n+"b"*n)
    return anbn


def make_anbn_continuation(w):
    length = len(w) - 1
    cont = [[0,1,1]] * int(length/2) + [[0,0,1]] * int(length/2)
    cont.insert(0,[1, 1, 0])
    cont[-1] = [1,0,0]
    return cont


def make_anbn_io_cont_redundant(N, p):
    anbn = gen_anbn_words_redundant(N, p)
    anbn.sort(key=len)
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


def make_anbn_io_cont_shuffled(N, p):
    anbn = gen_anbn_words_redundant(N, p)
    anbn = shuffler(anbn)
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
    cont = [[1, 1, 1]]+[[1, 1, 1]] * n + [[1, 0, 1]] * m
    return cont


def make_anbm_io_cont_redundant(N, p):
    anbm = gen_anbm_words_redundant(N, p)
    anbm.sort(key=len)
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


def make_anbm_io_cont_shuffled(N, p):
    anbm = gen_anbm_words_redundant(N, p)
    anbm = shuffler(anbm)
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


# (ab)n
def gen_abn_words_redundant(N, p):
    abn = []
    for i in range(N):
        n = geometric(p)
        abn.append("S"+"ab"*n)
    return abn


def make_abn_continuation(w):
    n = (len(w)-1)//2
    cont = [[1, 1, 0], [0, 0, 1]] * n + [[1, 1, 0]]
    return cont


def make_abn_io_cont_redundant(N, p):
    abn = gen_abn_words_redundant(N, p)
    abn.sort(key=len)
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


def make_abn_io_cont_shuffled(N, p):
    abn = gen_abn_words_redundant(N, p)
    abn = shuffler(abn)
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


# anbnan
def gen_anbnan_words_redundant(N, p):
    anbnan = []
    for i in range(N):
        n = geometric(p)
        anbnan.append("S"+"a"*n+"b"*n+"a"*n)
    return anbnan


def make_anbnan_continuation(w):
    n = (len(w)-1)//3
    cont = [[1, 1, 0]]+ [[0, 1, 1]] * n + [[0, 0, 1]] * (n-1) + [[0,1,0]]*n + [[1,0,0]]*(n!=0)
    return cont


def make_anbnan_io_cont_redundant(N, p):
    anbnan = gen_anbnan_words_redundant(N, p)
    anbnan.sort(key=len)
    x = []
    y = []
    mask = []
    for word in anbnan:
        x.append(torch.Tensor(list(map(dict_map, word))))
        y.append(torch.Tensor(make_anbnan_continuation(word)))
        mask.append(torch.ones(len(word)))
    x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True).type(torch.IntTensor)
    y = torch.nn.utils.rnn.pad_sequence(y, batch_first=True).type(torch.IntTensor)
    mask = torch.nn.utils.rnn.pad_sequence(mask, batch_first=True).type(torch.BoolTensor)

    return x, y, mask


def make_anbnan_io_cont_shuffled(N, p):
    anbnan = gen_anbnan_words_redundant(N, p)
    anbnan = shuffler(anbnan)
    x = []
    y = []
    mask = []
    for word in anbnan:
        x.append(torch.Tensor(list(map(dict_map, word))))
        y.append(torch.Tensor(make_anbnan_continuation(word)))
        mask.append(torch.ones(len(word)))
    x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True).type(torch.IntTensor)
    y = torch.nn.utils.rnn.pad_sequence(y, batch_first=True).type(torch.IntTensor)
    mask = torch.nn.utils.rnn.pad_sequence(mask, batch_first=True).type(torch.BoolTensor)

    return x, y, mask


# double dyck
def make_double_abplus_redundant(N, p):
    double_abplus = []
    for i in range(N):
        n = geometric(p)
        double_abplus.append("S" + ''.join(random.choice(["a","b"]) for _ in range(n))*2)
    return double_abplus


def make_double_abplus_continuation(w):
    cont = []
    for i, c in enumerate(w):
        if i % 2 == 0:
            if w[1:i//2+1] == w[i//2+1:i+1]:
                cont.append([1, 1, 1])
            else:
                cont.append([0, 1, 1])
        else:
            cont.append([0, 1, 1])
    return cont


def make_double_abplus_io_cont_redundant(N, p):
    double_abplus = make_double_abplus_redundant(N, p)
    double_abplus.sort(key=len)
    x = []
    y = []
    mask = []
    for word in double_abplus:
        x.append(torch.Tensor(list(map(dict_map, word))))
        y.append(torch.Tensor(make_double_abplus_continuation(word)))
        mask.append(torch.ones(len(word)))
    x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True).type(torch.IntTensor)
    y = torch.nn.utils.rnn.pad_sequence(y, batch_first=True).type(torch.IntTensor)
    mask = torch.nn.utils.rnn.pad_sequence(mask, batch_first=True).type(torch.BoolTensor)

    return x, y, mask


def make_double_abplus_io_cont_shuffled(N, p):
    double_abplus = make_double_abplus_redundant(N, p)
    double_abplus = shuffler(double_abplus)
    x = []
    y = []
    mask = []
    for word in double_abplus:
        x.append(torch.Tensor(list(map(dict_map, word))))
        y.append(torch.Tensor(make_double_abplus_continuation(word)))
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