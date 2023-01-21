from classifiers import *
import random
random.seed(12345)
import torch
torch.manual_seed(12345)
import numpy as np
import copy
np.random.seed(12345)
import os

def run_seq(num, models_dir, filename, generator_function, lambdas, epochs,  *model_args, **kwargs):
    if not os.path.isdir(models_dir):
        os.mkdir(models_dir)

    with open(filename + "_loss.csv", 'w', newline='') as accuracy_file:
        writer_details = csv.writer(accuracy_file)
        writer_details.writerow(["model_num", "lambda", "loss", "accuracy", "epoch", "l2", "l0", "re loss"])
        for lam in lambdas:
            for i in range(num):
                language_set = generator_function(**kwargs)
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
                writer_details.writerow([i + 1, lam, loss.item(), percent_correct.item(), epoch, l2, l0, re_loss])
                torch.save(model_out, "{}/model_{}_{}.pt".format(
                    models_dir, str(lam), str(i)))



from datetime import date
import os
N = 15
lambdas = [.00001, .000015, .00002, .000025, .00003, .000035, .00004, .000045, .00005] #, .01, .02, .03, .04, .05]
lambdas = lambdas+[i + .000045 for i in lambdas]
lambdas = lambdas+[i * 10 for i in lambdas]
new_dir = "Output-{}".format(str(date.today()))
if not os.path.isdir(new_dir):
    os.mkdir(new_dir)
models_dir = new_dir + "/models"
if not os.path.isdir(models_dir):
    os.mkdir(models_dir)

epochs = 1750

run_seq(N, "{}/l12_lstm".format(models_dir), "{}/l12_lstm".format(new_dir), lb.make_l12_sets, lambdas,
                   epochs,
                   *(5, 3, 8, 8, 4), **{"N": 1000, "p": .05, "reject_threshold": 200, "split_p": .795})