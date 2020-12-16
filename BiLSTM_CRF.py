# Author: Robert Guthrie

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import time
import pickle

torch.manual_seed(1)


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq: list, to_ix: dict):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def read_train_data(path: str) -> [([str], [str])]:
    train_data_list = []
    sentence = []
    states = []
    train_data = open(path, encoding='utf-8')
    for line in train_data:
        line = line.strip()
        if not line:
            if not sentence:
                print("two or more null lines!!")
                continue
            train_data_list.append((sentence, states))
            sentence = []
            states = []
        else:
            assert len(line) == 3, "len(line) != 3"
            sentence.append(line[0])
            states.append(line[2])
    train_data.close()
    print("finish read dataset, ", len(train_data_list), " lines totally")
    return train_data_list


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)  # (len(sentence), len(tag_to_ix))

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


def train():
    training_data = read_train_data(r"E:\大三上\智能系统\LAB2\dataset\dataset1\\train.utf8")[0:100]
    print(training_data)
    for sentence, tags in training_data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    global model
    model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    # Check predictions before training
    with torch.no_grad():
        precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
        precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
        print(model(precheck_sent))
    start = time.time()
    # Make sure prepare_sequence from earlier in the LSTM section is loaded
    for epoch in range(
            10):  # again, normally you would NOT do 300 epochs, it is toy data
        print("epoch", epoch)
        test()
        for sentence, tags in training_data:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is,
            # turn them into Tensors of word indices.
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

            # Step 3. Run our forward pass.
            loss = model.neg_log_likelihood(sentence_in, targets)

            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss.backward()
            optimizer.step()
    end = time.time()
    print("time:", (end - start) / 60, "min")
    # Check predictions after training
    test()
    with torch.no_grad():
        precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
        print(model(precheck_sent))
    # We got it!
    save_argument()


def save_argument():
    write_file = open(r'E:\大三上\智能系统\LAB2\lab2_submission\wordseg\LSTM argument\model.pkl', 'wb')
    pickle.dump(model, write_file, -1)
    write_file.close()
    write_file = open(r'E:\大三上\智能系统\LAB2\lab2_submission\wordseg\LSTM argument\dictionary.pkl', 'wb')
    pickle.dump(word_to_ix, write_file, -1)
    write_file.close()


def load_argument():
    global word_to_ix
    global model
    model_file = open(r'E:\大三上\智能系统\LAB2\lab2_submission\wordseg\LSTM argument\model.pkl', 'rb')
    model = pickle.load(model_file)
    model_file.close()
    model_file = open(r'E:\大三上\智能系统\LAB2\lab2_submission\wordseg\LSTM argument\dictionary.pkl', 'rb')
    word_to_ix = pickle.load(model_file)
    model_file.close()


def predict(sentence: str) -> str:
    sent_list = list(sentence)
    with torch.no_grad():
        precheck_sent = prepare_sequence(sent_list, word_to_ix)
        _, tag_list = model(precheck_sent)
    tag_str = "".join([ix_to_tag[tag_ix] for tag_ix in tag_list])
    return tag_str


def test() -> float:
    corr = 0
    total = 0
    for i in range(0, len(examples)):
        outputs = predict(examples[i])
        corr += sum([1 if a == b else 0 for a, b in zip(golds[i], outputs)])
        total += len(outputs)
        print("given:")
        print(segment(examples[i], golds[i]))
        print("predict:")
        print(segment(examples[i], outputs))
    # sen = "我爱北京天安门"
    # pre = predict(sen)
    # corr += sum([1 if a == b else 0 for a, b in zip("SSBEBIE", pre)])
    # total += len(pre)
    return corr / total


def segment(obs:str, states:str) -> str:
    segmented = ""
    assert len(obs) == len(states), "len(obs) != len(states)!"
    for i in range(0, len(obs)):
        segmented += obs[i]
        if states[i] == 'S' or states[i] == 'E':
            segmented += '/'
    return segmented


START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5
HIDDEN_DIM = 4

word_to_ix = {}
tag_to_ix = {"B": 0, "I": 1, "E": 2, START_TAG: 3, STOP_TAG: 4, "S": 5}
ix_to_tag = {0: "B", 1: "I", 2: "E", 3: START_TAG, 4: STOP_TAG, 5: "S"}
model = None

examples = open(r"E:\大三上\智能系统\LAB2\lab2_submission\example_dataset\input.utf8", encoding="utf8").readlines()
golds = open(r"E:\大三上\智能系统\LAB2\lab2_submission\example_dataset\gold.utf8", encoding="utf8").readlines()
examples = [ele.strip() for ele in examples]
golds = [ele.strip() for ele in golds]

if __name__ == "__main__":
    train()
    load_argument()
    test()