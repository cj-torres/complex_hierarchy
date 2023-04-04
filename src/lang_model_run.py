from classifiers import *
import random
random.seed(12345)
import torch
torch.manual_seed(12345)
import numpy as np
import copy
np.random.seed(12345)
import os

def lstm_intersection_test(num, models_dir, filename, generator_function, lambdas, epochs, test_langs, lang_names, *model_args, **kwargs):
    if not os.path.isdir(models_dir):
        os.mkdir(models_dir)

    with open(filename + "_loss.csv", 'w', newline='') as accuracy_file:
        ce_loss = torch.nn.CrossEntropyLoss()
        writer_details = csv.writer(accuracy_file)
        writer_details.writerow(["model num", "lambda", "loss", "accuracy", "epoch", "l2", "l0", "re loss",
                                 lang_names[0] + " loss", lang_names[1] + " loss"])
        for lam in lambdas:
            for i in range(num):
                language_set = generator_function(**kwargs)
                test_lang_sets = [test_lang(**kwargs) for test_lang in test_langs]
                print("Model %d" % (i + 1))
                model = LSTMSequencer(*model_args)
                for model_out, loss, percent_correct, epoch in seq_train(model, language_set, 256, epochs, 5,
                                                                                l0_regularized=True, lam=lam):
                    if loss.isnan().any():
                        break
                    # weights = model_to_list(model_out)
                with torch.no_grad():
                    l2 = model_out.count_l2().item()
                    l0 = model_out.count_l0().item()
                    re_loss = model_out.regularization().item()
                    tests = []
                    for test_lang in test_lang_sets:
                        model_out.eval()
                        x_test = test_lang.test_input.to("cuda")
                        y_test = test_lang.test_output.to("cuda")
                        mask_test = test_lang.test_mask.to("cuda")

                        y_test_hat = model_out(x_test)
                        loss_test = ce_loss(y_test_hat[mask_test], y_test[mask_test])
                        tests.append(loss_test.item())
                    print("{} loss: {} {} loss: {}".format(lang_names[0], tests[0], lang_names[1], tests[2]))
                writer_details.writerow([i + 1, lam, loss.item(), percent_correct.item(), epoch, l2, l0, re_loss, *tests])
                torch.save(model_out, "{}/model_{}_{}.pt".format(
                    models_dir, str(lam), str(i)))


def rnn_intersection_test(num, models_dir, filename, generator_function, lambdas, epochs, test_langs, lang_names, *model_args, **kwargs):
    if not os.path.isdir(models_dir):
        os.mkdir(models_dir)

    with open(filename + "_loss.csv", 'w', newline='') as accuracy_file:
        ce_loss = torch.nn.CrossEntropyLoss()
        writer_details = csv.writer(accuracy_file)
        writer_details.writerow(["model num", "lambda", "loss", "accuracy", "epoch", "l2", "l0", "re loss",
                                 lang_names[0] + " loss", lang_names[1] + " loss"])
        for lam in lambdas:
            for i in range(num):
                language_set = generator_function(**kwargs)
                test_lang_sets = [test_lang(**kwargs) for test_lang in test_langs]
                print("Model %d" % (i + 1))
                model = RNNSequencer(*model_args)
                for model_out, loss, percent_correct, epoch in seq_train(model, language_set, 256, epochs, 5,
                                                                                l0_regularized=True, lam=lam):
                    if loss.isnan().any():
                        break
                    # weights = model_to_list(model_out)
                with torch.no_grad():
                    l2 = model_out.count_l2().item()
                    l0 = model_out.count_l0().item()
                    re_loss = model_out.regularization().item()
                    tests = []
                    for test_lang in test_lang_sets:
                        model_out.eval()
                        x_test = test_lang.test_input.to("cuda")
                        y_test = test_lang.test_output.to("cuda")
                        mask_test = test_lang.test_mask.to("cuda")

                        y_test_hat = model_out(x_test)
                        loss_test = ce_loss(y_test_hat[mask_test], y_test[mask_test])
                        tests.append(loss_test.item())
                    print("{} loss: {} {} loss: {}".format(lang_names[0], tests[0], lang_names[1], tests[1]))
                writer_details.writerow([i + 1, lam, loss.item(), percent_correct.item(), epoch, l2, l0, re_loss, *tests])
                torch.save(model_out, "{}/model_{}_{}.pt".format(
                    models_dir, str(lam), str(i)))


def vib_intersection_test(num, models_dir, filename, generator_function, lambdas, epochs, test_langs, lang_names, *model_args, **kwargs):
    if not os.path.isdir(models_dir):
        os.mkdir(models_dir)

    with open(filename + "_loss.csv", 'w', newline='') as accuracy_file:
        writer_details = csv.writer(accuracy_file)
        writer_details.writerow(["model num", "lambda", "loss", "mi", "accuracy", "epoch",
                                 lang_names[0] + " loss", lang_names[0] + " mi",
                                 lang_names[1] + " loss", lang_names[1] + " mi"])
        for lam in lambdas:
            for i in range(num):
                language_set = generator_function(**kwargs)
                test_lang_sets = [test_lang(**kwargs) for test_lang in test_langs]
                print("Model %d" % (i + 1))
                model = vib.SequentialVariationalIB(*model_args)
                for model_out, loss, mi, percent_correct, epoch in vib_seq_train(model, language_set, 256, epochs, 5,
                                                                                 lam=lam):
                    if loss.isnan().any():
                        break
                    # weights = model_to_list(model_out)
                with torch.no_grad():
                    tests = []
                    for test_lang in test_lang_sets:
                        model_out.eval()
                        x_test = test_lang.test_input.to("cuda")
                        y_test = test_lang.test_output.to("cuda")
                        mask_test = test_lang.test_mask.to("cuda")

                        y_test_hat, stats, seq, probs,  = model_out(x_test)
                        loss_test = model.lm_loss(y_test, y_test_hat, mask_test)
                        tests.append(loss_test.item())
                        mi_test = model.mi_loss(seq, probs, mask_test)
                        tests.append(mi_test.item())
                    print("{} loss: {} {} loss: {}".format(lang_names[0], tests[0], lang_names[1], tests[1]))
                writer_details.writerow([i + 1, lam, loss.item(), mi.item(), percent_correct.item(), epoch, *tests])
                torch.save(model_out, "{}/model_{}_{}.pt".format(
                    models_dir, str(lam), str(i)))



from datetime import date
import os
N = 5
lambdas = [.00001, .00002, .00003, .00004, .00005, .00006, .00007, .00008, .00009] #, .01, .02, .03, .04, .05]
lambdas = [i*10 for i in lambdas]+ [i*100 for i in lambdas]+ [i*1000 for i in lambdas]
#lambdasib = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
#lambdasib = [i*.1 for i in lambdasib] + [i*.01 for i in lambdasib]
print(len(lambdas))
epochs = 1200

new_dir = "VIB-Output-{}".format(str(date.today()))
if not os.path.isdir(new_dir):
    os.mkdir(new_dir)
models_dir = new_dir + "/models"
if not os.path.isdir(models_dir):
    os.mkdir(models_dir)

vib_intersection_test(N, "{}/l13_lstm".format(models_dir), "{}/l13_lstm".format(new_dir), lb.make_l13_sets, lambdas,
                        epochs, [lb.make_l1_sets, lb.make_l3_sets], ["l1", "l3"],
                        *(5, 3, 5, 5, 5, 2, 2), **{"N": 2000, "p": .05, "reject_threshold": 200, "split_p": .795})

new_dir = "LSTM-Output-{}".format(str(date.today()))
if not os.path.isdir(new_dir):
    os.mkdir(new_dir)
models_dir = new_dir + "/models"
if not os.path.isdir(models_dir):
    os.mkdir(models_dir)

lstm_intersection_test(N, "{}/l13_lstm".format(models_dir), "{}/l13_lstm".format(new_dir), lb.make_l13_sets, lambdas,
                        epochs, [lb.make_l1_sets, lb.make_l3_sets], ["l1", "l3"],
                        *(5, 3, 5, 5), **{"N": 2000, "p": .05, "reject_threshold": 200, "split_p": .795})

new_dir = "RNN-Output-{}".format(str(date.today()))
if not os.path.isdir(new_dir):
    os.mkdir(new_dir)
models_dir = new_dir + "/models"
if not os.path.isdir(models_dir):
    os.mkdir(models_dir)

rnn_intersection_test(N, "{}/l13_lstm".format(models_dir), "{}/l13_lstm".format(new_dir), lb.make_l13_sets, lambdas,
                        epochs, [lb.make_l1_sets, lb.make_l3_sets], ["l1", "l3"],
                        *(5, 3, 5, 5), **{"N": 2000, "p": .05, "reject_threshold": 200, "split_p": .795})



#intersection_test(N, "{}/l12_lstm".format(models_dir), "{}/l12_lstm".format(new_dir), lb.make_l12_sets, lambdas,
#                   epochs, [lb.make_l1_sets, lb.make_l2_sets], ["l1", "l2"],
#                   *(5, 3, 5, 5, 4), **{"N": 2000, "p": .05, "reject_threshold": 200, "split_p": .795})

#intersection_test(N, "{}/l23_lstm".format(models_dir), "{}/l23_lstm".format(new_dir), lb.make_l23_sets, lambdas,
#                   epochs, [lb.make_l2_sets, lb.make_l3_sets], ["l2", "l3"],
#                   *(5, 3, 5, 5, 4), **{"N": 2000, "p": .05, "reject_threshold": 200, "split_p": .795})