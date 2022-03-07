from classifiers import *

if __name__ == '__main__':
    target_accuracies = [1]
    x, y, mask = lb.make_abn_io_cont_redundant(1000, .1)
    accuracy =  []
    with open('abn_model_c_lstm_seq1_rand.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for target in target_accuracies:
            for i in range(1500):
                print("Model %d" % (i+1))
                model = LSTMBranchSequencer(4, 2, 4, 4, 3)
                (x1, y1, lengths1), (x_t, y_t, lengths_t) = random_split(x, y, mask)
                model, best_loss, percent_correct = seq_train(model, x1, y1, lengths1, x_t, y_t, lengths_t, target, 500, 256)
                weights = model_to_list(model)
                writer.writerow(weights)
                accuracy.append([best_loss.item(), percent_correct.item(), target])
    with open('abn_model_c_lstm_loss_seq1_rand.csv', 'w', newline='') as accuracy_file:
        writer = csv.writer(accuracy_file)
        writer.writerow(["r", "accuracy", "target"])
        writer.writerows(accuracy)

    x, y, mask = lb.make_anbm_io_cont_redundant(1000, .1)
    accuracy = []
    with open('anbm_model_c_lstm_seq1_rand.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for target in target_accuracies:
            for i in range(1500):
                print("Model %d" % (i+1))
                model = LSTMBranchSequencer(4, 2, 4, 4, 3)
                (x1, y1, lengths1), (x_t, y_t, lengths_t) = random_split(x, y, mask)
                model, best_loss, percent_correct = seq_train(model, x1, y1, lengths1, x_t, y_t, lengths_t, target, 500, 256)
                weights = model_to_list(model)
                writer.writerow(weights)
                accuracy.append([best_loss.item(), percent_correct.item(), target])
    with open('anbm_model_c_lstm_loss_seq1_rand.csv', 'w', newline='') as accuracy_file:
        writer = csv.writer(accuracy_file)
        writer.writerow(["r", "accuracy", "target"])
        writer.writerows(accuracy)