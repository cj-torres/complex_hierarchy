from classifiers import *
import random
random.seed(12345)
import torch
torch.manual_seed(12345)
import numpy as np
np.random.seed(12345)


def intersection_run(num, lambdas, epochs, language_union_gen, lang_1_gen, lang_2_gen, *model_args, **lang_args):
    models_dir = "../data/Output-{}".format(str(date.today()))
    ce_loss = torch.nn.CrossEntropyLoss()
    if not os.path.isdir(models_dir):
        os.mkdir(models_dir)
    if not os.path.isdir(models_dir+"/models"):
        os.mkdir(models_dir+"/models")
    if not os.path.isdir(models_dir + "/lstm"):
        os.mkdir(models_dir + "/lstm")
        os.mkdir(models_dir + "/lstm/models")
    #if not os.path.isdir(models_dir+"/rnn"):
    #    os.mkdir(models_dir+"/rnn")
    #    os.mkdir(models_dir + "/rnn/models")
    #if not os.path.isdir(models_dir + "/vib"):
    #    os.mkdir(models_dir + "/vib")
    #    os.mkdir(models_dir + "/vib/models")

    best_lstm_l1 = lstm_best_fit(num*2, lang_1_gen, epochs, *model_args, **lang_args)
    best_lstm_l2 = lstm_best_fit(num*2, lang_2_gen, epochs, *model_args, **lang_args)
    best_lstm_union = lstm_best_fit(num*2, language_union_gen, epochs, *model_args, **lang_args)
    #best_rnn_l1 = rnn_best_fit(num*2, lang_1_gen, epochs, *model_args, **lang_args)
    #best_rnn_l2 = rnn_best_fit(num*2, lang_2_gen, epochs, *model_args, **lang_args)
    #best_rnn_union = rnn_best_fit(num*2, language_union_gen, epochs, *model_args, **lang_args)


    best_lstm_l1.to("cpu")
    best_lstm_l2.to("cpu")
    #best_rnn_l1.to("cpu")
    #best_rnn_l2.to("cpu")
    best_lstm_union.to("cpu")
    #best_rnn_union.to("cpu")

    torch.save(best_lstm_l1, models_dir + "/models/best_lstm_l1.pt")
    torch.save(best_lstm_l2, models_dir + "/models/best_lstm_l2.pt")
    #torch.save(best_rnn_l1, models_dir + "/models/best_rnn_l1.pt")
    #torch.save(best_rnn_l2, models_dir + "/models/best_rnn_l1.pt")
    torch.save(best_lstm_union, models_dir + "/models/best_lstm_l13.pt")
    #torch.save(best_rnn_union, models_dir + "/models/best_rnn_l13.pt")

    #vib_file = open(models_dir + "/vib/vib_loss.csv", 'w', newline='')
    lstm_file = open(models_dir + "/lstm/lstm_loss.csv", 'w', newline='')
    #rnn_file = open(models_dir + "/rnn/rnn_loss.csv", 'w', newline='')

    #vib_writer = csv.writer(vib_file)
    lstm_writer = csv.writer(lstm_file)
    #rnn_writer = csv.writer(rnn_file)

    #vib_writer.writerow(["num", "lam", "loss", "mi", "percent correct", "epoch", "loss lang 1", "mi lang 1",
    #                     "loss lang 2", "mi lang 2", "emp best lang 1", "emp best lang 2", "best lang 1", "best lang 2",
    #                     "empirical crossent 1", "empirical crossent 2"])
    lstm_writer.writerow(["num", "lam", "loss", "percent correct", "epoch", "l2", "l0", "re loss", "loss lang 1",
                          "loss lang 2", "emp best lang 1", "emp best lang 2", "best lang 1", "best lang 2",
                         "empirical crossent 1", "empirical crossent 2"])
    #rnn_writer.writerow(["num", "lam", "loss", "percent correct", "epoch", "l2", "l0", "re loss", "loss lang 1",
    #                      "loss lang 2", "emp best lang 1", "emp best lang 2", "best lang 1", "best lang 2",
    #                     "empirical crossent 1", "empirical crossent 2"])

    for j, lam in enumerate(lambdas):
        for i in range(num):
            language_set, lang_1, lang_2 = lb.make_intersection_complement_sets(lang_1_gen, lang_2_gen,
                                                                language_union_gen, 200, **lang_args)
            #lang_1 = lb.make_pfsa_sets(lang_1_gen, **lang_args)
            #lang_2 = lb.make_pfsa_sets(lang_2_gen, **lang_args)

            lstm_l1_empirical_best = ce_loss(best_lstm_l1(lang_1.test_input)[lang_1.test_mask],
                                               lang_1.test_output[lang_1.test_mask]).item()
            lstm_l2_empirical_best = ce_loss(best_lstm_l2(lang_2.test_input)[lang_2.test_mask],
                                               lang_2.test_output[lang_2.test_mask]).item()
            #rnn_l1_empirical_best = ce_loss(best_rnn_l1(lang_1.test_input)[lang_1.test_mask],
            #                                  lang_1.test_output[lang_1.test_mask]).item()
            #rnn_l2_empirical_best = ce_loss(best_rnn_l2(lang_2.test_input)[lang_2.test_mask],
            #                                  lang_2.test_output[lang_2.test_mask]).item()
            l1_theoretical_best = (lang_1.test_nll.sum() / lang_1.test_mask.sum()).item()
            l2_theoretical_best = (lang_2.test_nll.sum() / lang_2.test_mask.sum()).item()

            lstm_l1_empirical_crossent = ce_loss(best_lstm_union(lang_1.test_input)[lang_1.test_mask],
                                             lang_1.test_output[lang_1.test_mask]).item()
            lstm_l2_empirical_crossent = ce_loss(best_lstm_union(lang_2.test_input)[lang_2.test_mask],
                                             lang_2.test_output[lang_2.test_mask]).item()
            #rnn_l1_empirical_crossent = ce_loss(best_rnn_union(lang_1.test_input)[lang_1.test_mask],
            #                                lang_1.test_output[lang_1.test_mask]).item()
            #rnn_l2_empirical_crossent = ce_loss(best_rnn_union(lang_2.test_input)[lang_2.test_mask],
            #                                lang_2.test_output[lang_2.test_mask]).item()

            lstm_model, loss, percent_correct, epoch, l2, l0, re_loss, loss_lang_1, loss_lang_2 = lstm_intersection_run(
                lam, epochs, language_set, lang_1, lang_2, *model_args
            )

            lstm_writer.writerow([i+1, lam, loss, percent_correct, epoch, l2, l0, re_loss, loss_lang_1, loss_lang_2,
                                  lstm_l1_empirical_best, lstm_l2_empirical_best, l1_theoretical_best, l2_theoretical_best,
                                  lstm_l1_empirical_crossent, lstm_l2_empirical_crossent])

            torch.save(lstm_model, models_dir + "/lstm/models/lstm_{}_{}.pt".format(str(lam), str(i+1)))
            ####

            #rnn_model, loss, percent_correct, epoch, l2, l0, re_loss, loss_lang_1, loss_lang_2 = rnn_intersection_run(
            #    lam, epochs, language_set, lang_1, lang_2, *model_args
            #)

            #rnn_writer.writerow([i+1, lam, loss, percent_correct, epoch, l2, l0, re_loss, loss_lang_1, loss_lang_2,
            #                      rnn_l1_empirical_best, rnn_l2_empirical_best, l1_theoretical_best, l2_theoretical_best,
            #                     rnn_l1_empirical_crossent, rnn_l2_empirical_crossent])

            #torch.save(rnn_model, models_dir + "/rnn/models/rnn_{}_{}.pt".format(str(lam), str(i+1)))
            ####

            #vib_model, loss, mi, percent_correct, epoch, loss_lang_1, mi_lang_1, loss_lang_2, mi_lang_2 = vib_intersection_run(
            #    lam, epochs, language_set, lang_1, lang_2, *model_args
            #)

            #vib_writer.writerow([i+1, lam, loss, mi, percent_correct, epoch, loss_lang_1, mi_lang_1, loss_lang_2,
            #                     mi_lang_2, rnn_l1_empirical_best, rnn_l2_empirical_best, l1_theoretical_best, l2_theoretical_best,
            #                     rnn_l1_empirical_crossent, rnn_l2_empirical_crossent])

            #torch.save(vib_model, models_dir + "/vib/models/vib_{}_{}.pt".format(str(lam), str(i + 1)))

            ####


def language_runs(num, lambdas, epochs, languages, language_names, *model_args, **lang_args):
    models_dir = "../data/Output-{}".format(str(date.today()))
    if not os.path.isdir(models_dir):
        os.mkdir(models_dir)
    ce_loss = torch.nn.CrossEntropyLoss()

    for e, lang in enumerate(languages):
        if not os.path.isdir(models_dir + "/" + language_names[e]):
            os.mkdir(models_dir + "/" + language_names[e])
        if not os.path.isdir(models_dir + "/" + language_names[e] + "/models"):
            os.mkdir(models_dir + "/" + language_names[e] + "/models")
        if not os.path.isdir(models_dir + "/" + language_names[e] + "/lstm"):
            os.mkdir(models_dir + "/" + language_names[e] + "/lstm")
            os.mkdir(models_dir + "/" + language_names[e] + "/lstm/models")
        if not os.path.isdir(models_dir + "/" + language_names[e] + "/rnn"):
            os.mkdir(models_dir + "/" + language_names[e] + "/rnn")
            os.mkdir(models_dir + "/" + language_names[e] + "/rnn/models")
        #if not os.path.isdir(models_dir + "/" + language_names[e] + "/vib"):
        #    os.mkdir(models_dir + "/" + language_names[e] + "/vib")
        #    os.mkdir(models_dir + "/" + language_names[e] + "/vib/models")

        best_lstm = lstm_best_fit(num * 2, lang, epochs, *model_args, **lang_args)
        # best_lstm_union = lstm_best_fit(num*2, language_union_gen, epochs, *model_args, **lang_args)
        best_rnn = rnn_best_fit(num * 2, lang, epochs, *model_args, **lang_args)

        #vib_file = open(models_dir + "/" + language_names[e] + "/vib/vib_loss.csv", 'w', newline='')
        lstm_file = open(models_dir + "/" + language_names[e] + "/lstm/lstm_loss.csv", 'w', newline='')
        rnn_file = open(models_dir + "/" + language_names[e] + "/rnn/rnn_loss.csv", 'w', newline='')

        #vib_writer = csv.writer(vib_file)
        lstm_writer = csv.writer(lstm_file)
        rnn_writer = csv.writer(rnn_file)

        #vib_writer.writerow(["num", "lam", "dev loss", "mi", "percent correct", "epoch", "test loss", "test mi",
        #                     "emp best", "best"])
        lstm_writer.writerow(["num", "lam", "dev loss", "percent correct", "epoch", "l2", "l0", "re loss", "test loss",
                              "emp best", "best"])
        rnn_writer.writerow(["num", "lam", "dev loss", "percent correct", "epoch", "l2", "l0", "re loss", "test loss",
                             "emp best", "best"])

        best_lstm.to("cpu")
        best_rnn.to("cpu")

        torch.save(best_lstm, models_dir + "/" + language_names[e] +  "/models/best_lstm.pt")
        torch.save(best_rnn, models_dir + "/" + language_names[e] +  "/models/best_rnn.pt")

        for j, lam in enumerate(lambdas):
            for i in range(num):
                language_set = lb.make_pfsa_sets(lang, **lang_args)
                #test_set = lb.make_pfsa_sets(lang, **lang_args)

                lstm_empirical_best = ce_loss(best_lstm(language_set.test_input)[language_set.test_mask],
                                              language_set.test_output[language_set.test_mask]).item()
                rnn_empirical_best = ce_loss(best_rnn(language_set.test_input)[language_set.test_mask],
                                             language_set.test_output[language_set.test_mask]).item()

                theoretical_best = (language_set.test_nll.sum() / language_set.test_mask.sum()).item()

                lstm_model = None
                for lstm_model, loss, percent_correct, epoch, l2, l0, re_loss, test_loss in lstm_run(
                    lam, epochs, language_set, language_names[e], *model_args
                ):

                    lstm_writer.writerow([i + 1, lam, loss, percent_correct, epoch, l2, l0, re_loss, test_loss,
                                          lstm_empirical_best, theoretical_best])
                if lstm_model:
                    torch.save(lstm_model, models_dir + "/" + language_names[e] + "/lstm/models/lstm_{}_{}.pt".format(str(lam)[:8], str(i + 1)))
                ####

                rnn_model = None
                for rnn_model, loss, percent_correct, epoch, l2, l0, re_loss, test_loss in rnn_run(
                    lam, epochs, language_set, language_names[e], *model_args
                ):

                    rnn_writer.writerow([i + 1, lam, loss, percent_correct, epoch, l2, l0, re_loss, test_loss,
                                         rnn_empirical_best, theoretical_best])

                if rnn_model:
                    torch.save(rnn_model, models_dir + "/" + language_names[e] +"/rnn/models/rnn_{}_{}.pt".format(str(lam)[:8], str(i + 1)))
                ####

                # vib_model = None
                # for vib_model, loss, mi, percent_correct, epoch, test_loss, test_mi in vib_run(
                #     lam, epochs, language_set, language_names[e], *model_args
                # ):
                #
                #     vib_writer.writerow([i + 1, lam, loss, mi, percent_correct, epoch, test_loss, test_mi,
                #                          rnn_empirical_best, theoretical_best])
                # if vib_model:
                #     torch.save(vib_model, models_dir + "/" + language_names[e] +"/vib/models/vib_{}_{}.pt".format(str(lam)[:8], str(i + 1)))


def lstm_run(lam, epochs, language_set, lang_name, *model_args):
    ce_loss = torch.nn.CrossEntropyLoss()
    model = LSTMSequencer(*model_args)
    #old_output = (None, None, None, None)
    for output in seq_train(model, language_set, 256, epochs, 5, l0_regularized=True, lam=lam, patience=50):
        model_out, dev_loss, percent_correct, epoch = output
        if dev_loss.isnan().any():
            #output = old_output
            break
        #old_output = output
        #model_out, loss, percent_correct, epoch = output
        with torch.no_grad():
            l2 = model_out.count_l2().item()
            l0 = model_out.count_l0().item()
            re_loss = model_out.regularization().item()
            model_out.eval()
            x_lang = language_set.test_input.to("cuda")
            y_lang = language_set.test_output.to("cuda")
            mask_lang = language_set.test_mask.to("cuda")
            y_lang_hat = model_out(x_lang)
            test_loss = ce_loss(y_lang_hat[mask_lang], y_lang[mask_lang])

        yield model_out, dev_loss.item(), percent_correct.item(), epoch, l2, l0, re_loss, test_loss.item()
        print("{} loss: {}".format(lang_name, test_loss.item()))
    #return model_out, train_loss.item(), percent_correct.item(), epoch, l2, l0, re_loss, loss_lang.item()


def rnn_run(lam, epochs, language_set, lang_name, *model_args):
    ce_loss = torch.nn.CrossEntropyLoss()
    model = RNNSequencer(*model_args)
    #old_output = (None, None, None, None)
    for output in seq_train(model, language_set, 256, epochs, 5, l0_regularized=True, lam=lam, patience=50):
        model_out, dev_loss, percent_correct, epoch = output
        if dev_loss.isnan().any():
            #output = old_output
            break
        #old_output = output
        model_out, loss, percent_correct, epoch = output
        with torch.no_grad():
            l2 = model_out.count_l2().item()
            l0 = model_out.count_l0().item()
            re_loss = model_out.regularization().item()
            model_out.eval()
            x_lang = language_set.test_input.to("cuda")
            y_lang = language_set.test_output.to("cuda")
            mask_lang = language_set.test_mask.to("cuda")
            y_lang_hat = model_out(x_lang)
            test_loss = ce_loss(y_lang_hat[mask_lang], y_lang[mask_lang])


        yield model_out, dev_loss.item(), percent_correct.item(), epoch, l2, l0, re_loss, test_loss.item()
        print("{} loss: {}".format(lang_name, test_loss.item()))


def vib_run(lam, epochs, language_set, lang_name, *model_args):
    ce_loss = torch.nn.CrossEntropyLoss()
    model = vib.RecursiveGaussianRNN(*model_args)
    old_output = (None, None, None, None)
    for output in vib_seq_train(model, language_set, 256, epochs, 5, lam=lam, patience=50):
        model_out, dev_loss, mi, percent_correct, epoch = output
        if dev_loss.isnan().any():
            #output = old_output
            break
        #old_output = output
    #model_out, train_loss, mi, percent_correct, epoch = output
        with torch.no_grad():
            model_out.eval()

            x_lang = language_set.test_input.to("cuda")
            y_lang = language_set.test_output.to("cuda")
            mask_lang = language_set.test_mask.to("cuda")

            y_lang_hat, stats, seq, probs, = model_out(x_lang)
            test_loss = model.lm_loss(y_lang, y_lang_hat, mask_lang)
            mi_test = model.mi_loss(seq, probs, mask_lang)


        yield model_out, dev_loss.item(), mi.item(), percent_correct.item(), epoch, test_loss.item(), mi_test.item()
        print("{} loss: {}".format(lang_name, dev_loss.item()))



def lstm_intersection_run(lam, epochs, language_set, lang_1, lang_2, *model_args):
    ce_loss = torch.nn.CrossEntropyLoss()
    model = LSTMSequencer(*model_args)
    old_output = (None, None, None, None)
    for output in seq_train(model, language_set, 256, epochs, 5, l0_regularized=True, lam=lam, patience=50):
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
    for output in seq_train(model, language_set, 256, epochs, 5, l0_regularized=True, lam=lam, patience=50):
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
    for output in vib_seq_train(model, language_set, 256, epochs, 5, lam=lam, patience=50):
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
                                                                            l0_regularized=False, lam=1, patience=50):
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
                                                                            l0_regularized=False, lam=1, patience=50):
                if loss.isnan().any():
                    break
                # weights = model_to_list(model_out)
            writer_details.writerow([i + 1, loss.item(), percent_correct.item(), epoch])
            torch.save(model_out, "{}/model_{}.pt".format(
                models_dir, str(i)))


def rnn_best_fit(num, generator_function, epochs, *model_args, **kwargs):
    best_model = RNNSequencer(*model_args)
    best_performance = float("inf")
    language_set = lb.make_pfsa_sets(generator_function, **kwargs)
    for i in range(num):
        print("Model %d" % (i + 1))
        old_output = (None, None, None, None)
        model = RNNSequencer(*model_args)
        for output in seq_train(model, language_set, 256, epochs, 5, l0_regularized=False, lam=1, patience=50):
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
    language_set = lb.make_pfsa_sets(generator_function, **kwargs)
    for i in range(num):
        print("Model %d" % (i + 1))
        old_output = (None, None, None, None)
        model = LSTMSequencer(*model_args)
        for output in seq_train(model, language_set, 256, epochs, 5, l0_regularized=False, lam=1, patience=50):
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
#N=1
lambdas = torch.tensor(list(range(-6,-65,-4)))/10
lambdas = [10**x.item() for x in list(lambdas)]
#lambdas = [.00001, .00002, .00003, .00004, .00005, .00006, .00007, .00008, .00009]
#lambdas = [i*10 for i in lambdas] + [i*100 for i in lambdas] + [i*1000 for i in lambdas] + [i*10000 for i in lambdas]
#epochs = 20
epochs = 400
#lambdas = [.1]
languages = [lb.b1_pfsa, lb.b2_pfsa, lb.b3_pfsa, lb.g1_pfsa, lb.g2_pfsa, lb.g3_pfsa, lb.a1_pfsa, lb.a2_pfsa, lb.a3_pfsa]
names = ["b1", "b2", "b3", "g1", "g2", "g3", "a1", "a2", "a3"]

#languages = [lb.a1_pfsa, lb.a2_pfsa, lb.a3_pfsa, lb.b1_pfsa, lb.b2_pfsa, lb.b3_pfsa]
#names = ["a1", "a2", "a3", "b1", "b2", "b3"]


intersection_run(N, lambdas, epochs, lb.g13_pfsa, lb.g1_pfsa, lb.g3_pfsa,
                 *(5, 3, 5, 5), **{"n": 2000, "reject_threshold": 200, "split_p": .795})

#language_runs(N, lambdas, epochs, languages, names,
#              *(5, 3, 5, 5), **{"n": 2000, "reject_threshold": 200, "split_p": .795})





