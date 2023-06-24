from classifiers import *
import random
random.seed(12345)
import torch
torch.manual_seed(12345)
import numpy as np
import copy
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
                model = LSTMBranchSequencer(4, 2, 5, 5, 3)
                model, best_loss, percent_correct = seq_train(model, language_set, 256, 6000, 25)
                weights = model_to_list(model)
                writer.writerow(weights)
                accuracy.append([best_loss.item(), percent_correct.item(), target])
        with open(filename+"_loss.csv", 'w', newline='') as accuracy_file:
            writer = csv.writer(accuracy_file)
            writer.writerow(["best_loss", "accuracy", "target"])
            writer.writerows(accuracy)


def pre_train(language_set, model_template, desired_accuracy, epochs, *args, **kwargs):
    print("Pre-Training")
    fail = True
    while fail:
        model = model_template(*args, **kwargs)
        pre_train_gen = branch_seq_train(model, language_set, 512, epochs, 25, lam = 0)
        percent_correct = 0
        epoch = 0
        while percent_correct < desired_accuracy and epoch < epochs:
            model_out, _, percent_correct, epoch = next(pre_train_gen)
        if percent_correct >= desired_accuracy:
            fail = False
    return model_out


def regularized_branch(num, models_dir, filename, generator_function, lambdas, epochs,  *model_args, **kwargs):
    # runs trials of branch sequencers (tries to predict correct possible continuations)
    # branch sequencers regularized, uses list of lambdas as input
    loss_func = torch.nn.BCELoss()
    if not os.path.isdir(models_dir):
        os.mkdir(models_dir)

    with open(filename+"_loss.csv", 'w', newline='') as accuracy_file:
        writer_details = csv.writer(accuracy_file)
        writer_details.writerow(["model_num", "lambda", "loss", "accuracy", "epoch", "l2", "l0", "re loss", "test loss"])
        for lam in lambdas:
            for i in range(num):
                language_set = generator_function(**kwargs)
                print("Model %d" % (i + 1))
                model = LSTMBranchSequencer(*model_args)
                for model_out, loss, percent_correct, epoch in branch_seq_train(model, language_set, 256, epochs, 5,
                                                                                l0_regularized=True, lam=lam, patience=50):
                    if loss.isnan().any():
                        break
                    #weights = model_to_list(model_out)
                    if epoch % 5 == 0:
                        if epoch > 200 and (loss > .1 and percent_correct < 1):
                            torch.save(model_out, "{}/model_{}_{}.pt".format(
                                models_dir, str(lam), str(i)))

                    with torch.no_grad():
                        l2 = model_out.count_l2().item()
                        l0 = model_out.count_l0().item()
                        re_loss = model_out.regularization().item()

                        model_out.eval()
                        model_out.to("cuda")
                        x_test = language_set.test_input.to("cuda")
                        y_test = language_set.test_output.to("cuda")
                        mask_test = language_set.test_mask.to("cuda")

                        y_test_hat = model_out(x_test)
                        loss_test = loss_func(y_test_hat[mask_test], y_test[mask_test].to(torch.float))
                    writer_details.writerow([i + 1, lam, loss.item(), percent_correct.item(), epoch, l2, l0, re_loss, loss_test.item()])


def run_seq(num, filename, generator_function, weight_decay=False, **kwargs):
    with open(filename+".csv", 'w', newline='') as model_weight_file:
        with open(filename+"_loss.csv", 'w', newline='') as accuracy_file:
            writer_weights = csv.writer(model_weight_file)
            writer_details = csv.writer(accuracy_file)
            writer_details.writerow(["model_num", "best_loss", "accuracy", "epoch", "mdl"])
            for i in range(num):
                language_set = generator_function(**kwargs)
                print("Model %d" % (i + 1))
                model = LSTMSequencer(4, 2, 4, 4, 3)
                for model_out, best_loss, percent_correct, epoch in seq_train(model, language_set,
                                                                              512, 20, 10000, weight_decay):
                    weights = model_to_list(model_out)
                    writer_weights.writerow(weights)
                    encoding_cost = sum([1/2*w**2 for w in weights])
                    writer_details.writerow([i+1, best_loss.item(), percent_correct.item(), epoch, encoding_cost])


def rnn_regularized_branch(num, models_dir, filename, generator_function, lambdas, epochs,  *model_args, **kwargs):
    # runs trials of branch sequencers (tries to predict correct possible continuations)
    # branch sequencers regularized, uses list of lambdas as input
    loss_func = torch.nn.BCELoss()
    if not os.path.isdir(models_dir):
        os.mkdir(models_dir)

    with open(filename+"_loss.csv", 'w', newline='') as accuracy_file:
        writer_details = csv.writer(accuracy_file)
        writer_details.writerow(["model_num", "lambda", "loss", "accuracy", "epoch", "l2", "l0", "re loss", "test loss"])
        for lam in lambdas:
            for i in range(num):
                language_set = generator_function(**kwargs)
                print("Model %d" % (i + 1))
                model = RNNBranchSequencer(*model_args)
                for model_out, loss, percent_correct, epoch in branch_seq_train(model, language_set, 256, epochs, 5,
                                                                                l0_regularized=True, lam=lam, patience=50):
                    if loss.isnan().any():
                        break
                    #weights = model_to_list(model_out)
                    if epoch % 5 == 0:
                        if epoch > 50 and (loss > .1 and percent_correct < 1):
                            torch.save(model_out, "{}/model_{}_{}.pt".format(
                                models_dir, str(lam), str(i)))
                    with torch.no_grad():
                        l2 = model_out.count_l2().item()
                        l0 = model_out.count_l0().item()
                        re_loss = model_out.regularization().item()
                        model_out.eval()
                        model_out.to("cuda")
                        x_test = language_set.test_input.to("cuda")
                        y_test = language_set.test_output.to("cuda")
                        mask_test = language_set.test_mask.to("cuda")

                        y_test_hat = model_out(x_test)
                        loss_test = loss_func(y_test_hat[mask_test], y_test[mask_test].to(torch.float))

                    writer_details.writerow([i + 1, lam, loss.item(), percent_correct.item(), epoch, l2, l0, re_loss, loss_test.item()])


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
    N = 5

    lambdas = torch.tensor(list(range(-6, -65, -4))) / 10
    lambdas = list(reversed([10 ** x.item() for x in list(lambdas)]))
    #lambdas = [.00001, .000015, .00002, .000025, .00003, .000035, .00004, .000045, .00005] #, .01, .02, .03, .04, .05]
    #lambdas = lambdas+[i + .000045 for i in lambdas]
    new_dir = "Output-{}".format(str(date.today()))
    os.chdir("..\\data")
    if not os.path.isdir(new_dir):
        os.mkdir(new_dir)
    models_dir = new_dir + "/models"
    models_dir_rnn = new_dir + "/models/rnn"
    models_dir_lstm = new_dir + "/models/lstm"
    if not os.path.isdir(models_dir):
        os.mkdir(models_dir)
    if not os.path.isdir(models_dir_rnn):
        os.mkdir(models_dir_rnn)
    if not os.path.isdir(models_dir_lstm):
        os.mkdir(models_dir_lstm)

    epochs = 400

    regularized_branch(N, "{}/fl_small_lstm".format(models_dir_lstm), "{}/fl_small_lstm".format(new_dir),
                       lb.make_fl_branch_sets, lambdas,
                       epochs,
                       *(5, 3, 5, 5), **{"N": 2000, "p": .05, "reject_threshold": 200, "split_p": .795})
    regularized_branch(N, "{}/sh_small_lstm".format(models_dir_lstm), "{}/sh_small_lstm".format(new_dir),
                       lb.make_sh_branch_sets, lambdas,
                       epochs,
                       *(5, 3, 5, 5), **{"N": 2000, "p": .05, "reject_threshold": 200, "split_p": .795})

    #rnn_regularized_branch(N, "{}/fl_small_rnn".format(models_dir_rnn), "{}/fl_small_rnn".format(new_dir), lb.make_fl_branch_sets, lambdas,
    #                   epochs,
    #                   *(5, 3, 6, 6), **{"N": 1000, "p": .05, "reject_threshold": 200, "split_p": .795})
    #rnn_regularized_branch(N, "{}/sh_small_rnn".format(models_dir_rnn), "{}/sh_small_rnn".format(new_dir), lb.make_sh_branch_sets, lambdas,
    #                   epochs,
    #                   *(5, 3, 6, 6), **{"N": 1000, "p": .05, "reject_threshold": 200, "split_p": .795})



    #regularized_branch(N, "{}/dyck1_small_lstm".format(new_dir), lb.make_dyck1_sets_uniform_continuation, lambdas, epochs,
    #                   *(4, 2, 4, 4, 4), **{"N": 1000, "p": .05, "reject_threshold": 200, "split_p": .795})
    #regularized_branch(N, "{}/a2nb2m_small_lstm".format(new_dir), lb.make_a2nb2m_branch_sets, lambdas, epochs,
    #        **{"p": .05, "N": 1000, "reject_threshold": 200, "split_p": .795})
    #regularized_branch(N, "{}/abn_small_lstm".format(new_dir), lb.make_abn_branch_sets, lambdas, epochs,
    #        **{"p": .05, "N": 1000, "reject_threshold": 200, "split_p": .795})
    #regularized_branch(N, "{}/anbn_small_lstm".format(new_dir), lb.make_anbn_branch_sets, lambdas, epochs,
    #        **{"p": .05, "N": 1000, "reject_threshold": 200, "split_p": .795})













    #run_seq(N, "{}/dyck1_small_lstm_wd".format(new_dir), lb.make_dyck1_sets_uniform, weight_decay=True,
    #        **{"N": 1000, "p": .05, "reject_threshold": 200, "split_p": .795})
    #run_seq(N, "{}/a2nb2m_small_lstm_wd".format(new_dir), lb.make_a2nb2m_sets, weight_decay=True,
    #        **{"p": .05, "N": 1000, "reject_threshold": 200, "split_p": .795})




    # old runs
    # run_seq(N, "{}/dyck1_small_lstm".format(new_dir), lb.make_dyck1_sets_uniform,
    #         **{"N": 1000, "p": .05, "reject_threshold": 200, "split_p": .795})
    # run_seq(N, "{}/a2nb2m_small_lstm".format(new_dir), lb.make_a2nb2m_sets,
    #         **{"p":.05, "N": 1000, "reject_threshold": 200, "split_p": .795})
    # run_seq(N, "{}/abn_small_lstm".format(new_dir), lb.make_abn_sets,
    #         **{"p":.05, "N": 1000, "reject_threshold": 200, "split_p": .795})
    # run_seq(N, "{}/anbn_small_lstm".format(new_dir), lb.make_anbn_sets,
    #         **{"p":.05, "N": 1000, "reject_threshold": 200, "split_p": .795})
    # run_seq(N, "{}/dyck1_small_lstm_wd".format(new_dir), lb.make_dyck1_sets_uniform, weight_decay=True,
    #         **{"N": 1000, "p": .05, "reject_threshold": 200, "split_p": .795})
    # run_seq(N, "{}/a2nb2m_small_lstm_wd".format(new_dir), lb.make_a2nb2m_sets, weight_decay=True,
    #         **{"p": .05, "N": 1000, "reject_threshold": 200, "split_p": .795})
    # run_seq(N, "{}/abn_small_lstm_wd".format(new_dir), lb.make_abn_sets, weight_decay=True,
    #         **{"p": .05, "N": 1000, "reject_threshold": 200, "split_p": .795})
    # run_seq(N, "{}/anbn_small_lstm_wd".format(new_dir), lb.make_anbn_sets, weight_decay=True,
    #         **{"p": .05, "N": 1000, "reject_threshold": 200, "split_p": .795})
    # run_transf_seq(N, "{}/dyck1_small_trans".format(new_dir), lb.make_dyck1_sets_uniform,
    #         **{"N": 1000, "p": .05, "reject_threshold": 200, "split_p": .795})
    # run_transf_seq(N, "{}/a2nb2m_small_trans".format(new_dir), lb.make_a2nb2m_sets,
    #         **{"p": .05, "N": 1000, "reject_threshold": 200, "split_p": .795})
    # run_transf_seq(N, "{}/abn_small_trans".format(new_dir), lb.make_abn_sets,
    #         **{"p": .05, "N": 1000, "reject_threshold": 200, "split_p": .795})
    # run_transf_seq(N, "{}/anbn_small_trans".format(new_dir), lb.make_anbn_sets,
    #         **{"p": .05, "N": 1000, "reject_threshold": 200, "split_p": .795})
    #run(N, "anbnan", target_accuracies, lb.make_anbnan_io_cont_shuffled, **{"p": .1, "N": 1000})
    #run(N, "copy", target_accuracies, lb.make_double_abplus_io_cont_shuffled, **{"p": .1, "N": 1000})