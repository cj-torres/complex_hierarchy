from classifiers import *
import random
random.seed(12345)
import torch
torch.manual_seed(12345)
import numpy as np
np.random.seed(12345)


def full_run(num, lambdas, epochs, language_union_gen, lang_1_gen, lang_2_gen, *model_args, **lang_args):
    models_dir = "../data/Output-{}".format(str(date.today()))
    ce_loss = torch.nn.CrossEntropyLoss()
    if not os.path.isdir(models_dir):
        os.mkdir(models_dir)
    if not os.path.isdir(models_dir+"/models"):
        os.mkdir(models_dir+"/models")
    if not os.path.isdir(models_dir + "/lstm"):
        os.mkdir(models_dir + "/lstm")
        os.mkdir(models_dir + "/lstm/models")
    if not os.path.isdir(models_dir+"/rnn"):
        os.mkdir(models_dir+"/rnn")
        os.mkdir(models_dir + "/rnn/models")
    if not os.path.isdir(models_dir + "/vib"):
        os.mkdir(models_dir + "/vib")
        os.mkdir(models_dir + "/vib/models")

    best_lstm_l1 = lstm_best_fit(num, lang_1_gen, epochs, *model_args, **lang_args)
    best_lstm_l2 = lstm_best_fit(num, lang_2_gen, epochs, *model_args, **lang_args)
    best_rnn_l1 = rnn_best_fit(num, lang_1_gen, epochs, *model_args, **lang_args)
    best_rnn_l2 = rnn_best_fit(num, lang_2_gen, epochs, *model_args, **lang_args)

    best_lstm_l1.to("cpu")
    best_lstm_l2.to("cpu")
    best_rnn_l1.to("cpu")
    best_rnn_l2.to("cpu")

    torch.save(best_lstm_l1, models_dir + "/models/best_lstm_l1.pt")
    torch.save(best_lstm_l2, models_dir + "/models/best_lstm_l2.pt")
    torch.save(best_rnn_l1, models_dir + "/models/best_rnn_l1.pt")
    torch.save(best_rnn_l2, models_dir + "/models/best_rnn_l1.pt")

    vib_file = open(models_dir + "/vib/vib_loss.csv", 'w', newline='')
    lstm_file = open(models_dir + "/lstm/lstm_loss.csv", 'w', newline='')
    rnn_file = open(models_dir + "/rnn/rnn_loss.csv", 'w', newline='')

    vib_writer = csv.writer(vib_file)
    lstm_writer = csv.writer(lstm_file)
    rnn_writer = csv.writer(rnn_file)

    vib_writer.writerow(["num", "lam", "loss", "mi", "percent correct", "epoch", "loss lang 1", "mi lang 1",
                         "loss lang 2", "mi lang 2", "best lang 1", "best lang 2", "kld 1", "kld 2"])
    lstm_writer.writerow(["num", "lam", "loss", "percent correct", "epoch", "l2", "l0", "re loss", "loss lang 1",
                          "loss lang 2", "best lang 1", "best lang 2", "kld 1", "kld 2"])
    rnn_writer.writerow(["num", "lam", "loss", "percent correct", "epoch", "l2", "l0", "re loss", "loss lang 1",
                          "loss lang 2", "best lang 1", "best lang 2", "kld 1", "kld 2"])

    for j, lam in enumerate(lambdas):
        for i in range(num):
            language_set = language_union_gen(**lang_args)
            lang_1 = lang_1_gen(**lang_args)
            lang_2 = lang_2_gen(**lang_args)

            lstm_l1_theoretical_best = ce_loss(best_lstm_l1(lang_1.test_input)[lang_1.test_mask],
                                               lang_1.test_output[lang_1.test_mask]).item()
            lstm_l2_theoretical_best = ce_loss(best_lstm_l2(lang_2.test_input)[lang_2.test_mask],
                                               lang_2.test_output[lang_2.test_mask]).item()
            rnn_l1_theoretical_best = ce_loss(best_rnn_l1(lang_1.test_input)[lang_1.test_mask],
                                              lang_1.test_output[lang_1.test_mask]).item()
            rnn_l2_theoretical_best = ce_loss(best_rnn_l2(lang_2.test_input)[lang_2.test_mask],
                                              lang_2.test_output[lang_2.test_mask]).item()

            lstm_model, loss, percent_correct, epoch, l2, l0, re_loss, loss_lang_1, loss_lang_2 = lstm_intersection_run(
                lam, epochs, language_set, lang_1, lang_2, *model_args
            )

            lstm_writer.writerow([i+1, lam, loss, percent_correct, epoch, l2, l0, re_loss, loss_lang_1, loss_lang_2,
                                  lstm_l1_theoretical_best, lstm_l2_theoretical_best])

            torch.save(lstm_model, models_dir + "/lstm/models/lstm_{}_{}.pt".format(str(lam), str(i+1)))
            ####

            rnn_model, loss, percent_correct, epoch, l2, l0, re_loss, loss_lang_1, loss_lang_2 = rnn_intersection_run(
                lam, epochs, language_set, lang_1, lang_2, *model_args
            )

            rnn_writer.writerow([i+1, lam, loss, percent_correct, epoch, l2, l0, re_loss, loss_lang_1, loss_lang_2,
                                  rnn_l1_theoretical_best, rnn_l2_theoretical_best])

            torch.save(rnn_model, models_dir + "/rnn/models/rnn_{}_{}.pt".format(str(lam), str(i+1)))
            ####

            vib_model, loss, mi, percent_correct, epoch, loss_lang_1, mi_lang_1, loss_lang_2, mi_lang_2 = vib_intersection_run(
                lam, epochs, language_set, lang_1, lang_2, *model_args
            )

            vib_writer.writerow([i+1, lam, loss, mi, percent_correct, epoch, loss_lang_1, mi_lang_1, loss_lang_2,
                                 mi_lang_2, rnn_l1_theoretical_best, rnn_l2_theoretical_best])

            torch.save(vib_model, models_dir + "/vib/models/vib_{}_{}.pt".format(str(lam), str(i + 1)))

            ####



def lstm_intersection_run(lam, epochs, language_set, lang_1, lang_2, *model_args):
    ce_loss = torch.nn.CrossEntropyLoss()
    model = LSTMSequencer(*model_args)
    old_output = (None, None, None, None)
    for output in seq_train(model, language_set, 256, epochs, 5, l0_regularized=True, lam=lam, patience=20):
        model_out, loss, percent_correct, epoch = output
        if loss.isnan().any():
            output = old_output
            break
        old_output = output
    model_out, loss, percent_correct, epoch = output
    with torch.no_grad():
        l2 = model_out.count_l2().item()
        l0 = model_out.count_l0().item()
        re_loss = model_out.regularization().item()
        model_out.eval()
        x_lang_1 = lang_1.test_input.to("cuda")
        y_lang_1 = lang_1.test_output.to("cuda")
        mask_lang_1 = lang_1.test_mask.to("cuda")
        y_lang_1_hat = model_out(x_lang_1)
        loss_lang_1 = ce_loss(y_lang_1_hat[mask_lang_1], y_lang_1[mask_lang_1])

        x_lang_2 = lang_2.test_input.to("cuda")
        y_lang_2 = lang_2.test_output.to("cuda")
        mask_lang_2 = lang_2.test_mask.to("cuda")
        y_lang_2_hat = model_out(x_lang_2)
        loss_lang_2 = ce_loss(y_lang_2_hat[mask_lang_2], y_lang_2[mask_lang_2])

        print("G1 loss: {} G2 loss: {}".format(loss_lang_1.item(), loss_lang_2.item()))
    return model_out, loss.item(), percent_correct.item(), epoch, l2, l0, re_loss, loss_lang_1.item(), loss_lang_2.item()


def rnn_intersection_run(lam, epochs, language_set, lang_1, lang_2, *model_args):
    ce_loss = torch.nn.CrossEntropyLoss()
    model = RNNSequencer(*model_args)
    old_output = (None, None, None, None)
    for output in seq_train(model, language_set, 256, epochs, 5, l0_regularized=True, lam=lam, patience=20):
        model_out, loss, percent_correct, epoch = output
        if loss.isnan().any():
            output = old_output
            break
        old_output = output
    model_out, loss, percent_correct, epoch = output
    with torch.no_grad():
        l2 = model_out.count_l2().item()
        l0 = model_out.count_l0().item()
        re_loss = model_out.regularization().item()
        model_out.eval()
        x_lang_1 = lang_1.test_input.to("cuda")
        y_lang_1 = lang_1.test_output.to("cuda")
        mask_lang_1 = lang_1.test_mask.to("cuda")
        y_lang_1_hat = model_out(x_lang_1)
        loss_lang_1 = ce_loss(y_lang_1_hat[mask_lang_1], y_lang_1[mask_lang_1])

        x_lang_2 = lang_2.test_input.to("cuda")
        y_lang_2 = lang_2.test_output.to("cuda")
        mask_lang_2 = lang_2.test_mask.to("cuda")
        y_lang_2_hat = model_out(x_lang_2)
        loss_lang_2 = ce_loss(y_lang_2_hat[mask_lang_2], y_lang_2[mask_lang_2])

        print("G1 loss: {} G2 loss: {}".format(loss_lang_1.item(), loss_lang_2.item()))
    return model_out, loss.item(), percent_correct.item(), epoch, l2, l0, re_loss, loss_lang_1.item(), loss_lang_2.item()


def vib_intersection_run(lam, epochs, language_set, lang_1, lang_2, *model_args):
    model = vib.RecursiveGaussianRNN(*model_args)
    old_output = (None, None, None, None, None)
    for output in vib_seq_train(model, language_set, 256, epochs, 5, lam=lam, patience=20):
        model_out, loss, mi, percent_correct, epoch = output
        if loss.isnan().any():
            output = old_output
            break
        old_output = output
    model_out, loss, mi, percent_correct, epoch = output
                    # weights = model_to_list(model_out)
    model_out.eval()
    x_lang_1 = lang_1.test_input.to("cuda")
    y_lang_1 = lang_1.test_output.to("cuda")
    mask_lang_1 = lang_1.test_mask.to("cuda")

    y_lang_1_hat, stats, seq, probs,  = model_out(x_lang_1)
    loss_lang_1 = model.lm_loss(y_lang_1, y_lang_1_hat, mask_lang_1)
    mi_lang_1 = model.mi_loss(seq, probs, mask_lang_1)

    x_lang_2 = lang_2.test_input.to("cuda")
    y_lang_2 = lang_2.test_output.to("cuda")
    mask_lang_2 = lang_2.test_mask.to("cuda")

    y_lang_2_hat, stats, seq, probs, = model_out(x_lang_2)
    loss_lang_2 = model.lm_loss(y_lang_2, y_lang_2_hat, mask_lang_2)
    mi_lang_2 = model.mi_loss(seq, probs, mask_lang_2)

    print("G1 loss: {} G2 loss: {}".format(loss_lang_1.item(), loss_lang_2.item()))

    return model_out, loss.item(), mi.item(), percent_correct.item(), epoch, loss_lang_1.item(), mi_lang_1.item(),\
           loss_lang_2.item(), mi_lang_2.item()


def lstm_upper_bound(num, models_dir, filename, generator_function, epochs, *model_args, **kwargs):
    if not os.path.isdir(models_dir):
        os.mkdir(models_dir)

    with open(filename + "_loss.csv", 'w', newline='') as accuracy_file:
        writer_details = csv.writer(accuracy_file)
        writer_details.writerow(["model num", "loss", "accuracy", "epoch"])
        for i in range(num):
            language_set = generator_function(**kwargs)
            print("Model %d" % (i + 1))
            model = LSTMSequencer(*model_args)
            for model_out, loss, percent_correct, epoch in seq_train(model, language_set, 256, epochs, 5,
                                                                            l0_regularized=False, lam=1):
                if loss.isnan().any():
                    break
                # weights = model_to_list(model_out)
            writer_details.writerow([i + 1, loss.item(), percent_correct.item(), epoch])
            torch.save(model_out, "{}/model_{}.pt".format(
                models_dir, str(i)))


def rnn_upper_bound(num, models_dir, filename, generator_function, epochs, *model_args, **kwargs):
    if not os.path.isdir(models_dir):
        os.mkdir(models_dir)

    with open(filename + "_loss.csv", 'w', newline='') as accuracy_file:
        writer_details = csv.writer(accuracy_file)
        writer_details.writerow(["model num", "loss", "accuracy", "epoch"])
        for i in range(num):
            language_set = generator_function(**kwargs)
            print("Model %d" % (i + 1))
            model = RNNSequencer(*model_args)
            for model_out, loss, percent_correct, epoch in seq_train(model, language_set, 256, epochs, 5,
                                                                            l0_regularized=False, lam=1):
                if loss.isnan().any():
                    break
                # weights = model_to_list(model_out)
            writer_details.writerow([i + 1, loss.item(), percent_correct.item(), epoch])
            torch.save(model_out, "{}/model_{}.pt".format(
                models_dir, str(i)))


def rnn_best_fit(num, generator_function, epochs, *model_args, **kwargs):
    best_model = RNNSequencer(*model_args)
    best_performance = float("inf")
    language_set = generator_function(**kwargs)
    for i in range(num):
        print("Model %d" % (i + 1))
        old_output = (None, None, None, None)
        model = RNNSequencer(*model_args)
        for output in seq_train(model, language_set, 256, epochs, 5, l0_regularized=False, lam=1, patience=20):
            model_out, loss, percent_correct, epoch = output
            if loss.isnan().any():
                output = old_output
                break
            old_output = output
        model_out, loss, percent_correct, epoch = output
        if loss and loss < best_performance:
            best_performance = loss
            best_model = deepcopy(model_out)
    return best_model


def lstm_best_fit(num, generator_function, epochs, *model_args, **kwargs):
    best_model = LSTMSequencer(*model_args)
    best_performance = float("inf")
    language_set = generator_function(**kwargs)
    for i in range(num):
        print("Model %d" % (i + 1))
        old_output = (None, None, None, None)
        model = LSTMSequencer(*model_args)
        for output in seq_train(model, language_set, 256, epochs, 5, l0_regularized=False, lam=1, patience=20):
            model_out, loss, percent_correct, epoch = output
            if loss.isnan().any():
                output = old_output
                break
            old_output = output
        model_out, loss, percent_correct, epoch = output
        if loss and loss < best_performance:
            best_performance = loss
            best_model = deepcopy(model_out)
    return best_model


from datetime import date
import os
N = 5
lambdas = [.00001, .00002, .00003, .00004, .00005, .00006, .00007, .00008, .00009]
lambdas = [i*10 for i in lambdas] + [i*100 for i in lambdas] + [i*1000 for i in lambdas] + [i*10000 for i in lambdas]
epochs = 200


full_run(N, lambdas, epochs, lb.make_l13_sets, lb.make_l1_sets, lb.make_l3_sets,
         *(5, 3, 5, 5), **{"N": 2000, "p": .05, "reject_threshold": 400, "split_p": .795})

