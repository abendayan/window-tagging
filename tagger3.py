import sys
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import numpy as np
import time
import tagger as tagg
start_time = time.time()

if __name__ == '__main__':
    folder = sys.argv[1]
    # TODO make it optionnal
    vecs_file = sys.argv[2] if len(sys.argv) > 2 else False
    vocab_file = sys.argv[3] if len(sys.argv) > 3 else False
    tagger_training = tagg.Tagger(folder + '/train', True)
    # if we use the embedding words
    if vecs_file:
        name = ".embed."
        tagger_training.define_all_data_from_file(vecs_file, vocab_file)
    else:
        name = ""
        tagger_training.define_all_data()
    tagger_training.create_sub_words()
    tagger_training.activate_sub_words(tagger_training.vecs)
    tagger_dev = tagg.Tagger(folder + '/dev', True)
    tagg.model = tagg.Net(len(tagger_training.vocab), len(tagger_training.tags))
    tagg.model.embeddings.weight.data.copy_(torch.from_numpy(tagger_training.vecs))
    tagg.optimizer = optim.Adam(tagg.model.parameters(), lr=0.001)
    tagger_dev.define_x_y_not_train(tagger_training.vocab, tagger_training.tags_to_ix, True)
    tagger_test = tagg.Tagger(folder + '/test', True)
    tagger_test.define_x_y_not_train(tagger_training.vocab, tagger_training.tags_to_ix, False)

    losses = []
    epochs = 30
    data = []
    for epoch in range(epochs):
        print "Start epoch " + str(epoch),  str(tagg.passed_time(start_time))
        previous_time = time.time()
        acc_train, loss_train = tagger_training.learn(tagger_training, True)
        print "accuracy train: " + str(acc_train)
        print "loss train " + str(loss_train)
        print  "finish train epoch " + str(epoch) + " took " + str(tagg.passed_time(previous_time)) + "s"
        previous_time = time.time()
        acc_dev, loss_dev = tagger_dev.learn(tagger_training, False)
        print "accuracy dev: " + str(acc_dev)
        print "loss dev " + str(loss_dev)
        print  "finish dev epoch " + str(epoch) + " took " + str(tagg.passed_time(previous_time)) + "s"
        data.append([acc_train, loss_train, acc_dev, loss_dev])
        previous_time = time.time()
        file_values = tagger_test.learn(tagger_training, False, True)
        print  "finish test epoch " + str(epoch) + " took " + str(tagg.passed_time(previous_time)) + "s"
        previous_time = time.time()
        file_write = open(folder + '/test4.' + name + folder , 'w')
        file_write.write('\n'.join(file_values))
        file_write.close()
        print  "finish writing on file after epoch " + str(epoch) + " took " + str(tagg.passed_time(previous_time)) + "s"
    tagg.save_iter_in_graph(folder, data, "4" + name)
    print "saved in graph"
