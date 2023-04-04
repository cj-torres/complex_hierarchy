import torch
import csv
import language_builders as lb

model = torch.load("VIB-Output-2023-03-21\\models\\l13_lstm\\model_0.02_3.pt")

l13 = lb.make_l13_sets(**{"N": 2000, "p": .05, "reject_threshold": 200, "split_p": .795})

with open("pac_test.csv", 'w', newline='') as pca_file:
    model.eval()
