import sys
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import numpy as np
import time
import matplotlib.pyplot as plt

CONTEXT_SIZE = 5
EMBEDDING_DIM = 50
HIDDEN_DIM = 128
BATCH_SIZE = 512
torch.manual_seed(1)

start_time = time.time()

loss_function = nn.CrossEntropyLoss()
model = None
optimizer = None

def passed_time(previous_time):
    return round(time.time() - previous_time, 3)

def save_iter_in_graph(folder, data, add_to_name):
    graphes = {
        "test"+add_to_name+".accuracy_train": 0,
        "test"+add_to_name+".loss_train": 1,
        "test"+add_to_name+".accuracy_dev": 2,
        "test"+add_to_name+".loss_dev": 3
    }

    for graph, i in graphes.items():
        plt.figure(i)
        plt.plot(range(len(data)), [a[i] for a in data])
        plt.xlabel('Epochs')
        plt.ylabel(graph)
        plt.savefig(folder + '/' + graph + '.png')


class Net(nn.Module):
    def __init__(self, vocab_size, tag_size):
        super(Net, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, EMBEDDING_DIM)
        self.embeddings.shape = torch.Tensor(BATCH_SIZE, CONTEXT_SIZE * EMBEDDING_DIM)
        self.linear1 = nn.Linear(CONTEXT_SIZE * EMBEDDING_DIM, HIDDEN_DIM)
        self.dropout1 = nn.Dropout()
        self.linear2 = nn.Linear(HIDDEN_DIM, tag_size)
        self.dropout2 = nn.Dropout()

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view(self.embeddings.shape.size())
        out = F.tanh(self.linear1(self.dropout1(embeds)))
        out = self.linear2(self.dropout2(out))
        return out

class Tagger:
    def __init__(self, file_read, sub_word=False):
        # start by reading the file
        to_read = open(file_read, "r")
        self.vector_words = self.extract_words_from_file(to_read.read().split('\n'))
        self.windows = self.define_windows(self.vector_words)
        self.vocab = None
        self.vocab_to_ix = None
        self.tags = None
        self.windows_ix = None
        self.tags_to_ix = None
        self.tags_list = None
        self.x = None
        self.y = None
        self.vocab_to_add = None
        self.vecs_to_add = None
        self.sub_word = sub_word
        print "create object " + file_read + " after " + str(passed_time(start_time))

    def define_all_data(self):
        self.vocab = set([b[0] for b in self.vector_words])
        self.vocab.add('UUUNKKK')
        self.vocab_list = list(self.vocab)
        self.vocab_to_ix = {word: i for i, word in enumerate(self.vocab)}
        self.tags = set([b[1] for b in self.windows])
        self.tags_list = list(self.tags)
        self.tags_to_ix = {word: i for i, word in enumerate(self.tags)}
        self.windows_to_ix()
        # define the vecs
        self.vecs = [np.zeros(EMBEDDING_DIM)] * len(self.vocab)
        print "define windows done after " + str(passed_time(start_time))

    def define_all_data_from_file(self, vector_file, vocab_file):
        self.vecs = np.loadtxt(vector_file)
        vocab_txt = np.array(open(vocab_file, 'r').read().split('\n'))
        self.vocab = set([w for w in vocab_txt if w != ""])
        self.vocab_list = list(self.vocab)
        self.vocab_to_ix = {word: i for i, word in enumerate(self.vocab)}
        self.tags = set([b[1] for b in self.windows])
        self.tags_list = list(self.tags)
        self.tags_to_ix = {word: i for i, word in enumerate(self.tags)}
        self.windows_to_ix()
        print "define windows done after " + str(passed_time(start_time))

    def create_sub_words(self):
        print "create the sub words " + str(passed_time(start_time))
        index_unk = self.vocab_list.index('UUUNKKK')
        # the vectors for all of the prefix and suffix
        new_vec = np.array(self.vecs[index_unk])
        self.vocab_to_add = []
        self.vecs_to_add = []
        for word in self.vocab:
            self.vocab_to_add.append("@" + word[:3])
            self.vecs_to_add.append(new_vec)
            self.vocab_to_add.append(word[-3:] + "@")
            self.vecs_to_add.append(new_vec)

    def activate_sub_words(self, vecs):
        print "activate the subwords " + str(passed_time(start_time))
        for i, word in enumerate(self.vocab):
            vec = self.vecs[i]
            if "@" + word[:3] in self.vocab_to_add:
                index = self.vocab_to_add.index("@" + word[:3])
                vec = [x+y for x,y in zip(vec, self.vecs_to_add[index])]
            if word[-3:] + "@" in self.vocab_to_add:
                index = self.vocab_to_add.index(word[-3:] + "@")
                vec = [x+y for x,y in zip(vec, self.vecs_to_add[index])]
            self.vecs[i] = vec
        self.vecs = np.array(self.vecs)

    def word_or_unk(self, word):
        if word in self.vocab:
            return word
        else:
            if word.lower() in self.vocab:
                return word.lower()
            return 'UUUNKKK'

    def windows_to_ix(self):
        print "start in windows_to_ix " + str(passed_time(start_time))
        windows = []
        for window in self.windows:
            words, tag = window
            words_ix = []
            for word in words:
                words_ix.append(self.vocab_to_ix[self.word_or_unk(word)])
            windows.append((words_ix, self.tags_to_ix[tag]))

        self.windows_ix = windows
        print "start padding " + str(passed_time(start_time))
        while len(self.windows_ix) % BATCH_SIZE != 0:
            self.windows_ix.append([[0]*CONTEXT_SIZE, 0])
        print "finsih padding " + str(passed_time(start_time))
        self.x = torch.LongTensor([x for x, y in self.windows_ix])
        self.y = torch.LongTensor([y for x, y in self.windows_ix])
        train = Data.TensorDataset(self.x, self.y)
        self.train_loader = Data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)

    def define_x_y_not_train(self, vocab, tags_to_ix, shuffle_data):
        # the vocab is from the training file
        self.vocab = vocab
        self.vocab_text = [b[0] for b in self.vector_words]
        self.vocab_to_ix = {word: i for i, word in enumerate(vocab)}
        self.vocab_list = list(self.vocab)

        windows = []
        for window in self.windows:
            words, tag = window
            words_ix = []
            for word in words:
                words_ix.append(self.vocab_to_ix[self.word_or_unk(word)])
            if tag != '':
                windows.append((words_ix, tags_to_ix[tag]))
            else:
                windows.append((words_ix, 0))

        self.windows_ix = windows
        print "start padding " + str(passed_time(start_time))
        while len(self.windows_ix) % BATCH_SIZE != 0:
            self.windows_ix.append([[0]*CONTEXT_SIZE, 0])
        print "finsih padding " + str(passed_time(start_time))
        self.x = torch.LongTensor([x for x, y in self.windows_ix])
        self.y = torch.LongTensor([y for x, y in self.windows_ix])
        train = Data.TensorDataset(self.x, self.y)
        self.train_loader = Data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=shuffle_data)

    def get_ix_from_words(self, words):
        window = []
        for word in words:
            word = self.word_or_unk(word)
            window.append(self.vocab_list[word])
        return window

    def get_words_from_ix(self, words_ix):
        window = []
        for word_ix in words_ix:
            window.append(self.vocab_list[word_ix])
        return window

    def learn(self, train_tagger, is_train, is_test=False):
        checked = 0
        correct = 0
        total_loss = torch.Tensor([0])
        file_values = []

        for i, (b_x, b_y) in enumerate(self.train_loader):
            batch_x = autograd.Variable(b_x)
            batch_y = autograd.Variable(b_y)
            tags_window = b_y.numpy()
            words_window = b_x.numpy()
            model.zero_grad()

            forward = model(batch_x)

            prediction = torch.max(forward, -1)[1]
            checked += BATCH_SIZE
            if is_test:
                for j, y in enumerate(map(int, prediction[:BATCH_SIZE])):
                    tag = train_tagger.tags_list[y]
                    index_middle = int(CONTEXT_SIZE/2)
                    window = self.get_words_from_ix(words_window[j])

                    word = self.vector_words[j+i+index_middle][0]
                    if word != "</s>" and word != "<s>":
                        file_values.append(word + " " + tag)
                        if window[-index_middle:] == ["</s>", "</s>"]:
                            file_values.append('')
            else:
                checked -= len([1 for i, y in enumerate(map(int, prediction)) if (y == tags_window[i]) and train_tagger.tags_list[y] == 'O'])
                correct += len([1 for i, y in enumerate(map(int, prediction)) if (y == tags_window[i]) and train_tagger.tags_list[y] != 'O'])

            # compute the loss function
            loss = loss_function(forward, batch_y)

            if is_train:
                # backward step, only for training
                loss.backward()
                optimizer.step()
            total_loss += loss.data


        accuracy = round(float(correct) / checked, 3)
        loss = round(total_loss[0] / BATCH_SIZE, 3)
        if is_test:
            return file_values
        else:
            return accuracy, loss

    def define_data(self, vocab_to_ix, windows, tags_to_ix):
        X = []
        Y = []
        for window in windows:
            x = []
            words, tag = window
            word_ix = []
            for word in words:
                if word not in vocab_to_ix:
                    word = 'UUUNKKK'
                word_ix.append(vocab_to_ix[word])
            x.append(word_ix)
            Y.append(tags_to_ix[tag])
            X.append(x)
        return X, Y

    def define_windows(self, words_with_tags):
        middle = int(CONTEXT_SIZE/2)
        return [(list(map(lambda b: b[0], words_with_tags[i:i + CONTEXT_SIZE])), words_with_tags[i + middle][1]) for i in
                       range(len(words_with_tags) - CONTEXT_SIZE) if words_with_tags[i + middle] != 's']

    def extract_words_from_file(self, content_file):
        words_with_tags = [('<s>', 's'), ('<s>', 's')] # we start with an oppening of a phrase
        for i, word_tag in enumerate(content_file):
            if word_tag == "": # we are at the end of a sentence
                words_with_tags.append(('</s>', 's'))
                words_with_tags.append(('</s>', 's'))
                words_with_tags.append(('<s>', 's'))
                words_with_tags.append(('<s>', 's'))
            else:
                seperate = word_tag.split()
                if len(seperate) > 1:
                    # in the case of the training file
                    word = seperate[0]
                    tag = seperate[1]
                else:
                    word = seperate[0]
                    tag = ""
                words_with_tags.append((word, tag))
        words_with_tags.append(('</s>', 's'))
        words_with_tags.append(('</s>', 's'))
        return words_with_tags
