from classifiers import *
import random
random.seed(12345)
import torch
torch.manual_seed(12345)
import numpy as np
np.random.seed(12345)


def run_branch(num, filename, target_accuracies, generator_function, **kwargs):
    # runs trials of branch sequencers (tries to predict correct possible continuations)
    # must accept branch sequencer language set
    with open(filename+".csv", 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        accuracy = []
        for target in target_accuracies:
            for i in range(num):
                language_set = generator_function(**kwargs)
                print("Model %d" % (i + 1))
                model = LSTMBranchSequencer(4, 2, 4, 4, 3)
                model, best_loss, percent_correct = seq_train(model, language_set, target, False, 256)
                weights = model_to_list(model)
                writer.writerow(weights)
                accuracy.append([best_loss.item(), percent_correct.item(), target])
        with open(filename+"_loss.csv", 'w', newline='') as accuracy_file:
            writer = csv.writer(accuracy_file)
            writer.writerow(["best_loss", "accuracy", "target"])
            writer.writerows(accuracy)


def run_seq(num, filename, generator_function, **kwargs):
    with open(filename+".csv", 'w', newline='') as model_weight_file:
        with open(filename+"_loss.csv", 'w', newline='') as accuracy_file:
            writer_weights = csv.writer(model_weight_file)
            writer_details = csv.writer(accuracy_file)
            writer_details.writerow(["model_num", "best_loss", "accuracy", "target", "epoch"])
            for i in range(num):
                language_set = generator_function(**kwargs)
                print("Model %d" % (i + 1))
                model = LSTMSequencer(4, 2, 4, 4, 3)
                for model_out, best_loss, percent_correct, epoch in seq_train(model, language_set, 512, 20, 20):
                    weights = model_to_list(model_out)
                    writer_weights.writerow(weights)
                    writer_details.writerow([i+1, best_loss.item(), percent_correct.item(), epoch])

if __name__ == '__main__':
    target_accuracies = [.975]
    N = 1000
    run_seq(N, "dyck1_new", lb.make_dyck1_sets,
            **{"max_length": 200, "N": 1000, "reject_threshold": 200, "split_p": .795, "p": 50 / 151, "q": 50 / 151})
    run_seq(N, "anbm_new", lb.make_anbm_sets,
            **{"p":.01, "N":1000, "reject_threshold": 200, "split_p": .795})
    run_seq(N, "abn_new", lb.make_abn_sets,
            **{"p":.01, "N":1000, "reject_threshold": 200, "split_p": .795})
    run_seq(N, "anbn_new", lb.make_anbn_sets,
            **{"p":.01, "N":1000, "reject_threshold": 200, "split_p": .795})
    #run(N, "anbnan", target_accuracies, lb.make_anbnan_io_cont_shuffled, **{"p": .1, "N": 1000})
    #run(N, "copy", target_accuracies, lb.make_double_abplus_io_cont_shuffled, **{"p": .1, "N": 1000})