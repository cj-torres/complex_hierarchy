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
            writer_details.writerow(["model_num", "best_loss", "accuracy", "epoch", "mdl"])
            for i in range(num):
                language_set = generator_function(**kwargs)
                print("Model %d" % (i + 1))
                model = LSTMSequencer(4, 2, 4, 4, 3)
                for model_out, best_loss, percent_correct, epoch in seq_train(model, language_set, 512, 20, 10000):
                    weights = model_to_list(model_out)
                    writer_weights.writerow(weights)
                    encoding_cost = sum([1/2*w**2 for w in weights])
                    writer_details.writerow([i+1, best_loss.item(), percent_correct.item(), epoch, encoding_cost])


def run_transf_seq(num, filename, generator_function, **kwargs):
    with open(filename+".csv", 'w', newline='') as model_weight_file:
        with open(filename+"_loss.csv", 'w', newline='') as accuracy_file:
            writer_weights = csv.writer(model_weight_file)
            writer_details = csv.writer(accuracy_file)
            writer_details.writerow(["model_num", "best_loss", "accuracy", "target", "epoch", "mdl"])
            for i in range(num):
                language_set = generator_function(**kwargs)
                print("Model %d" % (i + 1))
                model = TransformerSequencer(4, 4, 2, 1, 4, 3)
                for model_out, best_loss, percent_correct, epoch in seq_train(model, language_set, 128, 20, 10000):
                    weights = model_to_list(model_out)
                    writer_weights.writerow(weights)
                    encoding_cost = sum([1 / 2 * w ** 2 for w in weights])
                    writer_details.writerow([i+1, best_loss.item(), percent_correct.item(), epoch, encoding_cost])
                    del model_out


if __name__ == '__main__':
    from datetime import date
    import os
    N = 100
    new_dir = "Output-{}".format(str(date.today()))
    os.mkdir(new_dir)
    #run_seq(N, "{}/dyck1_small_lstm".format(new_dir), lb.make_dyck1_sets_uniform,
    #        **{"N": 1000, "p": .01, "reject_threshold": 200, "split_p": .795})
    #run_seq(N, "{}/a2nb2m_small_lstm".format(new_dir), lb.make_a2nb2m_sets,
    #        **{"p":.01, "N": 1000, "reject_threshold": 200, "split_p": .795})
    #run_seq(N, "{}/abn_small_lstm".format(new_dir), lb.make_abn_sets,
    #        **{"p":.01, "N": 1000, "reject_threshold": 200, "split_p": .795})
    #run_seq(N, "{}/anbn_small_lstm".format(new_dir), lb.make_anbn_sets,
    #        **{"p":.01, "N": 1000, "reject_threshold": 200, "split_p": .795})
    run_transf_seq(N, "{}/dyck1_small_trans".format(new_dir), lb.make_dyck1_sets_uniform,
            **{"N": 1000, "p": .01, "reject_threshold": 200, "split_p": .795})
    run_transf_seq(N, "{}/a2nb2m_small_trans".format(new_dir), lb.make_a2nb2m_sets,
            **{"p": .01, "N": 1000, "reject_threshold": 200, "split_p": .795})
    run_transf_seq(N, "{}/abn_small_trans".format(new_dir), lb.make_abn_sets,
            **{"p": .01, "N": 1000, "reject_threshold": 200, "split_p": .795})
    run_transf_seq(N, "{}/anbn_small_trans".format(new_dir), lb.make_anbn_sets,
            **{"p": .01, "N": 1000, "reject_threshold": 200, "split_p": .795})
    #run(N, "anbnan", target_accuracies, lb.make_anbnan_io_cont_shuffled, **{"p": .1, "N": 1000})
    #run(N, "copy", target_accuracies, lb.make_double_abplus_io_cont_shuffled, **{"p": .1, "N": 1000})