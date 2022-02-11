from classifiers import *

if __name__ == '__main__':
    target_accuracies = [.5, .55, .6, .65, .7, .75, .8, .85, .9, .95]
    x, y, mask = lb.make_dyck1_io_cont(2000)
    accuracy =  []
    with open('dyck1_model_c_lstm_seq1.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for target in target_accuracies:
            for i in range(5000):
                print("Model %d" % (i+1))
                model = LSTMBranchSequencer(4, 2, 4, 4, 3)
                (x1, y1, lengths1), (x_t, y_t, lengths_t) = random_split(x, y, mask)
                model, best_loss, percent_correct = seq_train(model, x1, y1, lengths1, x_t, y_t, lengths_t, target, 500, 256)
                weights = model_to_list(model)
                writer.writerow(weights)
                accuracy.append([best_loss.item(), percent_correct.item()])
    with open('dyck1_model_c_lstm_loss_seq1.csv', 'w', newline='') as accuracy_file:
        writer = csv.writer(accuracy_file)
        writer.writerow(["r", "accuracy"])
        writer.writerows([[a] for a in accuracy])

    x, y, mask = lb.make_anbn_io_cont(2000)
    accuracy = []
    with open('anbn_model_c_lstm_seq1.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for target in target_accuracies:
            for i in range(5000):
                print("Model %d" % (i+1))
                model = LSTMBranchSequencer(4, 2, 4, 4, 3)
                (x1, y1, lengths1), (x_t, y_t, lengths_t) = random_split(x, y, mask)
                model, best_loss, percent_correct = seq_train(model, x1, y1, lengths1, x_t, y_t, lengths_t, target, 500, 256)
                weights = model_to_list(model)
                writer.writerow(weights)
                accuracy.append([best_loss.item(), percent_correct.item()])
    with open('anbn_model_c_lstm_loss_seq1.csv', 'w', newline='') as accuracy_file:
        writer = csv.writer(accuracy_file)
        writer.writerow(["r", "accuracy"])
        writer.writerows([[a] for a in accuracy])