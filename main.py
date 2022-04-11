from classifiers import *
import random
random.seed(12345)
import torch
torch.manual_seed(12345)
import numpy as np
np.random.seed(12345)


def run(num, filename, target_accuracies, generator_function, **kwargs):
    with open(filename+".csv", 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        accuracy = []
        for target in target_accuracies:
            for i in range(num):
                x, y, mask = generator_function(**kwargs)
                print("Model %d" % (i + 1))
                model = LSTMBranchSequencer(4, 2, 4, 4, 3)
                (x1, y1, lengths1), (x_t, y_t, lengths_t) = random_split_no_overfit(x, y, mask)
                model, best_loss, percent_correct = seq_train(model, x1, y1, lengths1, x_t, y_t, lengths_t, target, 500,
                                                              256)
                weights = model_to_list(model)
                writer.writerow(weights)
                accuracy.append([best_loss.item(), percent_correct.item(), target])
        with open(filename+"_loss.csv", 'w', newline='') as accuracy_file:
            writer = csv.writer(accuracy_file)
            writer.writerow(["best_loss", "accuracy", "target"])
            writer.writerows(accuracy)


if __name__ == '__main__':
    target_accuracies = [2, 1, .5, .25]
    N = 2500
    run(N,"anbm",target_accuracies,lb.make_anbm_io_cont_shuffled, **{"p":.1, "N":1000})
    run(N,"abn",target_accuracies,lb.make_abn_io_cont_shuffled, **{"p":.1, "N":1000})
    run(N,"anbn",target_accuracies,lb.make_anbn_io_cont_shuffled, **{"p":.1, "N":1000})
    run(N,"dyck1",target_accuracies,lb.make_dyck1_io_cont_shuffled, **{"max_length":200, "N":1000})
    run(N, "anbnan", target_accuracies, lb.make_anbnan_io_cont_shuffled, **{"p": .1, "N": 1000})
    run(N, "copy", target_accuracies, lb.make_double_abplus_io_cont_shuffled, **{"p": .1, "N": 1000})