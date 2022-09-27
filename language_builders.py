import random
import torch
import numpy as np
from collections import Counter
from dataclasses import dataclass


@dataclass
class LanguageData:
    train_input: torch.Tensor
    train_output: torch.Tensor
    train_mask: torch.Tensor
    test_input: torch.Tensor
    test_output: torch.Tensor
    test_mask: torch.Tensor


def dict_map(s):
    abcd_map = {'S': 1, 'a': 2, 'b': 3, 'c': 4, 'd': 5}
    return abcd_map[s]


def dyck_dict_map(s):
    int_map = {1: 2, -1: 3}
    return int_map[s]


def geometric(p):
    return np.random.geometric(p)


def to_tensors(set, dict_map = dict_map):
    input = []
    output = []
    mask = []
    for word in set:
        input.append(torch.Tensor(list(map(dict_map, word))))
        output.append(torch.Tensor(list(map(dict_map, word[1:]+"S"))))
        mask.append(torch.ones(len(word)))
    input = torch.nn.utils.rnn.pad_sequence(input, batch_first=True).type(torch.LongTensor)
    output = torch.nn.utils.rnn.pad_sequence(output, batch_first=True).type(torch.LongTensor)
    mask = torch.nn.utils.rnn.pad_sequence(mask, batch_first=True).type(torch.BoolTensor)

    return input, output, mask

def to_tensors_duchon(set):
    # for tensors generated with Duchon algorithm (already mapped)
    input = []
    output = []
    mask = []
    for word in set:
        input.append(word)
        output.append(torch.cat([word[1:], torch.Tensor([1])]))
        mask.append(torch.ones(len(word)))
    input = torch.nn.utils.rnn.pad_sequence(input, batch_first=True).type(torch.LongTensor)
    output = torch.nn.utils.rnn.pad_sequence(output, batch_first=True).type(torch.LongTensor)
    mask = torch.nn.utils.rnn.pad_sequence(mask, batch_first=True).type(torch.BoolTensor)

    return input, output, mask

def to_continuation_tensors(set, continuation_function, dict_map = dict_map):
    input = []
    output = []
    mask = []
    for word in set:
        input.append(torch.Tensor(list(map(dict_map, word))))
        output.append(torch.Tensor(continuation_function(word)))
        mask.append(torch.ones(len(word)))

    input = torch.nn.utils.rnn.pad_sequence(input, batch_first=True).type(torch.LongTensor)
    output = torch.nn.utils.rnn.pad_sequence(output, batch_first=True).type(torch.LongTensor)
    mask = torch.nn.utils.rnn.pad_sequence(mask, batch_first=True).type(torch.BoolTensor)

    return input, output, mask


def shuffler(language, split_percent, reject_size = 150):
    # returns training and testing sets with no overlap to prevent overfitting
    # because there are significant repeats a simple split won't work
    # here the language is lumped into word counts
    # counts are then shuffled then unpacked again into the set size
    # finally the test set is checked above a rejection threshold to throw away splits with small test set sizes
    N = len(language)
    split_count = N*split_percent
    lang_counts = Counter(language)
    shuffled = list(lang_counts.keys())
    shuffled_lang_train = []
    shuffled_lang_test = []
    while len(shuffled_lang_test) < reject_size:
        shuffled_lang_train = []
        shuffled_lang_test = []
        random.shuffle(shuffled)
        n = 0
        for key in shuffled:
            if n<split_count:
                n += lang_counts[key]
                shuffled_lang_train += [key for n in range(lang_counts[key])]
            else:
                shuffled_lang_test += [key for n in range(lang_counts[key])]

    assert (len(shuffled_lang_train)+len(shuffled_lang_test) == N)
    return shuffled_lang_train, shuffled_lang_test

### Language checkers, good for troubleshooting ###

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



### CONTEXT FREE LANGUAGES ###

###
# Dyck generating functions
#
#
#
###


def dyck1_transition(s, p=.5, q=.25):
    # for generation as a CFG
    # p and q represent probabilities of different rewrites
    # p: S -> aSb
    # q: S -> SS
    # 1-p-q: S -> e
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


def gen_dyck1_word(max_length, p=.5, q=.25):
    # calls transition function until:
    # all characters are terminal
    # or terminal characters exceed max_length
    a = "S"
    while "S" in a:
        a = dyck1_transition(a, p=p, q=q)
        if len(a.replace("S", "")) >= max_length:
            a = "S"
    return "S"+a


def dyck1_transition_geom(s, q=.5):
    # for generation as a CFG
    # p and q represent probabilities of different rewrites
    # p: S -> aSb
    # q: S -> SS
    # 1-p-q: S -> e
    assert (0 <= q <= 1)
    c = random.choice([i for i in range(len(s)) if s[i] == "S"])
    x = random.uniform(0, 1)

    return s[:c] + (x < q) * "aSb" + (q <= x) * "SS" + s[c + 1:]


def gen_dyck1_word_geom(l, q=.5):
    assert (0 <= q <= 1)
    word = "S"
    while len(word.replace("S", "")) < l*2:
        word = dyck1_transition_geom(word, q)
    return "S"+word.replace("S", "")


def gen_dyck1_words_redundant(N, max_length,  p=.5, q=.25):
    # generates N words in dyck1 not exceeding max_length
    # p and q supplied to lower functions
    dyck1 = []
    while len(dyck1) < N:
        dyck1.append(gen_dyck1_word(max_length, p=p, q=q))
    return dyck1


def gen_dyck1_words_geom(N, p,  q=.5):
    # generates N words in dyck1 not exceeding max_length
    # p and q supplied to lower functions
    dyck1 = []
    while len(dyck1) < N:
        dyck1.append(gen_dyck1_word_geom(geometric(p), q))
    return dyck1


def make_dyck1_sets(N, max_length, split_p, reject_threshold, p=.5, q=.25):
    # generates N words in dyck1 not exceeding max_length
    # p and q supplied to lower functions
    # split_p is the training set percentage (approximated), reject_threshold minimum test_set size
    # for converging lengths of mean mu set p and q s.t. mu = (mu+2)p+2*mu*q
    # p == q -> p = q = mu/(3*mu+2)
    assert(split_p <= 1)
    assert(N*(1-split_p) >= reject_threshold)
    dyck1 = gen_dyck1_words_redundant(N, max_length,  p=p, q=q)
    dyck1_train_in, dyck1_test_in = shuffler(dyck1, split_p, reject_threshold)
    dyck1_train = to_tensors(dyck1_train_in)
    dyck1_test = to_tensors(dyck1_test_in)

    return LanguageData(*dyck1_train+dyck1_test)


def make_dyck1_sets_geom(N, p, split_p, reject_threshold, q=.5):
    # generates N words in dyck1 not exceeding max_length
    # p and q supplied to lower functions
    # split_p is the training set percentage (approximated), reject_threshold minimum test_set size
    # for converging lengths of mean mu set p and q s.t. mu = (mu+2)p+2*mu*q
    # p == q -> p = q = mu/(3*mu+2)
    assert(split_p <= 1)
    assert(N*(1-split_p) >= reject_threshold)
    dyck1 = gen_dyck1_words_geom(N, p,  q)
    dyck1_train_in, dyck1_test_in = shuffler(dyck1, split_p, reject_threshold)
    dyck1_train = to_tensors(dyck1_train_in)
    dyck1_test = to_tensors(dyck1_test_in)

    return LanguageData(*dyck1_train+dyck1_test)


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


def make_dyck1_branch_sets(N, max_length, split_p, reject_threshold, p=.5, q=.25):
    dyck1 = gen_dyck1_words_redundant(N, max_length, p=p, q=q)
    dyck1_train_in, dyck1_test_in = shuffler(dyck1, split_p, reject_threshold)
    dyck1_train = to_continuation_tensors(dyck1_train_in, make_dyck1_continuation)
    dyck1_test = to_continuation_tensors(dyck1_test_in, make_dyck1_continuation)

    return LanguageData(*dyck1_train+dyck1_test)


### Duchon sampler ###
# Generates uniformly distributed Dyck words of a given length
# Overview of algorithm:
# 1. Randomly generate a balanced string (by shuffling an equal amount of a's and b's)
# 2. Check if string is a Dyck string
# 3. If not, for all "a"/1/"open-bracket" pivot word around "a" and check if Dyck string
#
# This algorithm works because the pivoting function defines an equivalence class on balanced strings
# which is guaranteed to contain one and only one Dyck word.
### Duchon 2000 ###

def dyck_seed(n):
    # initializes vector with an equal number of a's and b's (1, -1)
    return np.concatenate([np.ones(n), -np.ones(n)]) #.type(torch.LongTensor)


def torch_shuffle(array):
    # shuffles vector to sample random "Balanced" word
    return np.copy(np.random.permutation(array))


# to-do implement Zipfian distribution
# ensure lengths follow same distributions across languages

def dyck_sampler(n):
    # implements Duchon sampling above
    # matrices used for performance gain
    word = dyck_seed(n)
    word = torch_shuffle(word)

    sum_matrix = np.tril(np.ones((n*2, n*2)))

    if ((sum_matrix @ word) >= 0).all():
        # checks that all prefixes of word never have more closing than opening brackets
        # (or more -1 than 1)
        return torch.Tensor(np.insert(np.array([dyck_dict_map(s) for s in word]), 0, 1, axis=0))
        # returns words in corrected form to align with other formats

    # loop over word and pivot on a's/1's/open brackets
    for i, u in enumerate(word):
        if u == 1:
            candidate = np.concatenate([word[i+1:], word[i:i+1], word[:i]])
            # check if Dyck word and return if so
            if ((sum_matrix @ candidate) >= 0).all():
                return torch.Tensor(np.insert(np.array([dyck_dict_map(s) for s in candidate]), 0, 1, axis=0))


def gen_dyck_uniform(N, p):
    dyck = []
    for n in range(N):
        dyck.append(dyck_sampler(geometric(p)))

    return dyck


def make_dyck1_sets_uniform(N, p, split_p, reject_threshold):
    assert (split_p <= 1)
    assert (N * (1 - split_p) >= reject_threshold)
    dyck1 = gen_dyck_uniform(N, p)
    dyck1_train_in, dyck1_test_in = shuffler(dyck1, split_p, reject_threshold)
    dyck1_train = to_tensors_duchon(dyck1_train_in)
    dyck1_test = to_tensors_duchon(dyck1_test_in)

    return LanguageData(*dyck1_train + dyck1_test)

###
# a^n b^n generating functions
#
#
#
###


def gen_anbn_words_redundant(N, p):
    anbn = []
    for i in range(N):
        n = geometric(p)
        anbn.append("S"+"a"*n+"b"*n)
    return anbn


def make_anbn_sets(N, p, split_p, reject_threshold):
    # generates N words in anbn with n sampled from a geometric distribution
    # p geometric probability
    # mean of geometric is 1/p
    assert(split_p <= 1)
    assert(N*(1-split_p) >= reject_threshold)
    anbn = gen_anbn_words_redundant(N, p)
    anbn_train_in, anbn_test_in = shuffler(anbn, split_p, reject_threshold)
    anbn_train = to_tensors(anbn_train_in)
    anbn_test = to_tensors(anbn_test_in)

    return LanguageData(*anbn_train+anbn_test)


def make_anbn_continuation(w):
    length = len(w) - 1
    cont = [[0,1,1]] * int(length/2) + [[0,0,1]] * int(length/2)
    cont.insert(0,[1, 1, 0])
    cont[-1] = [1,0,0]
    return cont


def make_anbn_branch_sets(N, p, split_p, reject_threshold):
    anbn = gen_anbn_words_redundant(N, p)
    anbn_train_in, anbn_test_in = shuffler(anbn, split_p, reject_threshold)
    anbn_train = to_continuation_tensors(anbn_train_in, make_anbn_continuation)
    anbn_test = to_continuation_tensors(anbn_test_in, make_anbn_continuation)

    return LanguageData(*anbn_train+anbn_test)

### REGULAR LANGUAGES ###

###
# a^n b^m generating functions
#
#
#
###


def gen_anbm_words_redundant(N, p):
    anbm = []
    for i in range(N):
        n = geometric(p)
        m = geometric(p)
        anbm.append("S"+"a"*n+"b"*m)
    return anbm


def make_anbm_sets(N, p, split_p, reject_threshold):
    # generates N words in anbm with n and m sampled from a geometric distribution
    # p geometric probability
    # mean of geometric is 1/p
    assert(split_p <= 1)
    assert(N*(1-split_p) >= reject_threshold)
    anbm = gen_anbm_words_redundant(N, p)
    anbm_train_in, anbm_test_in = shuffler(anbm, split_p, reject_threshold)
    anbm_train = to_tensors(anbm_train_in)
    anbm_test = to_tensors(anbm_test_in)

    return LanguageData(*anbm_train+anbm_test)


def make_anbm_continuation(w):
    length = len(w)-1
    n = len(w.replace("b", ""))-1
    m = length - n
    cont = [[1, 1, 1]]+[[1, 1, 1]] * n + [[1, 0, 1]] * m
    return cont


def make_anbm_branch_sets(N, p, split_p, reject_threshold):
    anbm = gen_anbm_words_redundant(N, p)
    anbm_train_in, anbm_test_in = shuffler(anbm, split_p, reject_threshold)
    anbm_train = to_continuation_tensors(anbm_train_in, make_anbm_continuation)
    anbm_test = to_continuation_tensors(anbm_test_in, make_anbm_continuation)

    return LanguageData(*anbm_train+anbm_test)


###
# a^2n b^2m
# This language is to help make the lengths distributable as the other languages
#
###
def gen_a2nb2m_words_redundant(N, p):
    anbm = []
    for i in range(N):
        l = geometric(p)
        n = random.randint(0, l)
        m = l - n
        anbm.append("S"+"a"*2*n+"b"*2*m)
    return anbm


def make_a2nb2m_sets(N, p, split_p, reject_threshold):
    # generates N words in anbm with n and m sampled from a geometric distribution
    # p geometric probability
    # mean of geometric is 1/p
    assert(split_p <= 1)
    assert(N*(1-split_p) >= reject_threshold)
    anbm = gen_a2nb2m_words_redundant(N, p)
    anbm_train_in, anbm_test_in = shuffler(anbm, split_p, reject_threshold)
    anbm_train = to_tensors(anbm_train_in)
    anbm_test = to_tensors(anbm_test_in)

    return LanguageData(*anbm_train+anbm_test)





###
# ab^n generating functions
#
#
#
###


def gen_abn_words_redundant(N, p):
    abn = []
    for i in range(N):
        n = geometric(p)
        abn.append("S"+"ab"*n)
    return abn


def make_abn_sets(N, p, split_p, reject_threshold):
    # generates N words in anbn with n sampled from a geometric distribution
    # p geometric probability
    # mean of geometric is 1/p
    assert(split_p <= 1)
    assert(N*(1-split_p) >= reject_threshold)
    abn = gen_abn_words_redundant(N, p)
    abn_train_in, abn_test_in = shuffler(abn, split_p, reject_threshold)
    abn_train = to_tensors(abn_train_in)
    abn_test = to_tensors(abn_test_in)

    return LanguageData(*abn_train+abn_test)


def make_abn_continuation(w):
    n = (len(w)-1)//2
    cont = [[1, 1, 0], [0, 0, 1]] * n + [[1, 1, 0]]
    return cont


def make_abn_branch_sets(N, p, split_p, reject_threshold):
    abn = gen_abn_words_redundant(N, p)
    abn_train_in, abn_test_in = shuffler(abn, split_p, reject_threshold)
    abn_train = to_continuation_tensors(abn_train_in, make_abn_continuation)
    abn_test = to_continuation_tensors(abn_test_in, make_abn_continuation)

    return LanguageData(*abn_train+abn_test)





