import matplotlib.pyplot as plt
import numpy as np
import time

TRAIN_DATA1 = "E:\大三上\智能系统\LAB2\dataset\dataset1\\train.utf8"
TRAIN_DATA2 = "E:\大三上\智能系统\LAB2\dataset\dataset2\\train.utf8"
LABELS = "E:\大三上\智能系统\LAB2\dataset\dataset1\labels.utf8"
TEMPLATES = "E:\源程序\PycharmProjects\lab2\\template.utf8"
VALID_DATA = "E:\大三上\智能系统\LAB2\lab2_submission\example_dataset\input.utf8"
GOLD = "E:\大三上\智能系统\LAB2\lab2_submission\example_dataset\gold.utf8"

unigram_template = []
bigram_template = []
state_list = []
weights_dic = {}
epochs = 5
examples = open(r"E:\大三上\智能系统\LAB2\lab2_submission\example_dataset\input.utf8", encoding="utf8").readlines()
golds = open("E:\大三上\智能系统\LAB2\lab2_submission\example_dataset\gold.utf8", encoding="utf8").readlines()
examples = [ele.strip() for ele in examples]
golds = [ele.strip() for ele in golds]


def init():
    # read templates
    tf = open(TEMPLATES, encoding='utf-8')
    for template_str in tf:
        if template_str[0] == '#':
            continue
        start_index = 0
        template_num = []
        #  找到每一个[]里面的第一个数字
        while start_index != -1:
            start_index = template_str.find("[", start_index + 1)
            if start_index != -1:
                end_index = template_str.find(",", start_index)
                template_num.append(int(template_str[start_index + 1: end_index]))
        if len(template_num) > 0:
            # 将这一行的特征模板加入templates
            if template_str[0] == 'U':
                # Unigram
                unigram_template.append(template_num)
            elif template_str[0] == 'B':
                # bigram
                bigram_template.append(template_num)

    # read states
    label_file = open(LABELS)
    for state in label_file:
        state_list.append(state.strip())

    # print initial arguments
    print(state_list)
    print(unigram_template)
    print(bigram_template)
    print(weights_dic)


#  根据当前已有的特征进行预测，训练和分词的时候都要用到
def viterbi(sentence:str) -> str:
    assert len(sentence) > 0, "null str"
    V = [{}]
    Path = []
    for key in state_list:
        # 因为Path的第一个没有前面，即本来应该是Path[0][key] = null，这里直接不赋值，后面就在赋值Path的时候位置是t-1
        V[0][key] = get_uni_weights(sentence, 0, key) + get_bi_weights(sentence, 0, key, " ")  # 0前面没有字符，但是还要占一个位置，所以传" "而不是""
    for t in range(1, len(sentence)):
        V.append({})
        Path.append({})
        for key in state_list:
            (weight, last_state) = max(
                [(V[t - 1][y0] + get_uni_weights(sentence, t, key)
                  + get_bi_weights(sentence, t, key, y0), y0) for y0 in state_list])
            V[t][key] = weight
            Path[t - 1][key] = last_state  # 令state(t)为key的权重最大的state(t-1)
    _, max_state = max([(V[len(sentence) - 1][y], y) for y in state_list])
    state_str = max_state
    for t in range(len(sentence) - 2, -1, -1):
        state_str = Path[t][max_state] + state_str
        max_state = Path[t][max_state]  # 令当前状态的概率最大的上一个状态
    return state_str


def get_uni_weights(sentence:str, curr_pos:int, curr_state:str) -> int:
    weight = 0
    for i in range(0, len(unigram_template)):
        feature_key = generate_feature_key(unigram_template[i], i, sentence, curr_pos, curr_state)
        weight += weights_dic.get(feature_key, 0)
    return weight


def get_bi_weights(sentence:str, curr_pos:int, curr_state:str, prev_state:str) -> int:
    weight = 0
    for i in range(0, len(bigram_template)):
        feature_key = generate_feature_key(unigram_template[i], i, sentence, curr_pos, prev_state + curr_state)
        weight += weights_dic.get(feature_key, 0)
    return weight


def generate_feature_key(template_num:[int], identity:int, sentence:str, curr_pos:int, states:str) -> str:
    key = str(identity)
    for offset in template_num:
        index = curr_pos + offset
        if index < 0 or index >= len(sentence):
            key += " "
        else:
            key += sentence[index]
    return key + "/" + states


# 在一个句子上进行训练
def train_a_sentence(sentence:str, ref_states:str) -> (int, str):
    wrong_num = 0
    pred_states = viterbi(sentence)
    assert len(pred_states) == len(sentence), "len(pred_states) != len(sentence)" + "\n" + sentence + "\n" + pred_states
    # print(sentence)
    # print(segment(sentence, pred_states))
    for i in range(0, len(pred_states)):
        ref_state = ref_states[i]
        pred_state = pred_states[i]
        if ref_state != pred_state:
            wrong_num += 1
            # update unigram_template, wrong predict --, ref ++
            for t in range(0, len(unigram_template)):
                feature_key = generate_feature_key(unigram_template[t], t, sentence, i, pred_state)
                if feature_key in weights_dic:
                    weights_dic[feature_key] -= 1
                else:
                    weights_dic[feature_key] = -1
                feature_key = generate_feature_key(unigram_template[t], t, sentence, i, ref_state)
                if feature_key in weights_dic:
                    weights_dic[feature_key] += 1
                else:
                    weights_dic[feature_key] = 1

            # update bigram_template, wrong predict --, ref ++
            for t in range(0, len(bigram_template)):
                feature_key = generate_feature_key(unigram_template[t], t, sentence, i,
                                                   " " if i == 0 else pred_states[i-1] + pred_state)
                if feature_key in weights_dic:
                    weights_dic[feature_key] -= 1
                else:
                    weights_dic[feature_key] = -1
                feature_key = generate_feature_key(unigram_template[t], t, sentence, i,
                                                   " " if i == 0 else ref_states[i-1] + ref_state)
                if feature_key in weights_dic:
                    weights_dic[feature_key] += 1
                else:
                    weights_dic[feature_key] = 1
    return wrong_num, pred_states


def train():
    # (sentences, references) = read_dataset(TRAIN_DATA1)
    (sentences, references) = read_dataset(TRAIN_DATA2)
    # 两个dataset一起用
    # (s2, r2) = read_dataset(TRAIN_DATA2)
    # sentences[len(sentences):len(sentences)] = s2
    # references[len(references):len(references)] = r2
    assert len(sentences) == len(references), "len(sentences) != len(references)!!"
    train_accs = []
    valid_accs = []
    start = time.time()
    for epoch in range(0, epochs):
        wrong_num = 0
        total_test = 0
        for i in range(0, len(sentences)):
            sentence = sentences[i]
            states = references[i]
            assert len(sentence) == len(states), "len(sentence) != len(ref_states)!!"
            total_test += len(sentence)
            n, predict_states = train_a_sentence(sentence, states)
            wrong_num += n
            # if i % 1000 == 0:
            #     print("第",i,"行")
            #     print(sentence)
            #     print("given:")
            #     print(segment(sentence, states))
            #     print("predict")
            #     print(segment(sentence, predict_states))
        ta = 1 - (wrong_num/total_test)
        train_accs.append(ta)
        va = test()
        valid_accs.append(va)
        print("epoch ", epoch, "train accuracy=", ta, "valid accuracy=", va)
    end = time.time()
    print("finish training, time:", (end-start)/60, "min")
    xx = np.linspace(0, epochs-1, epochs)
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("CRF accuracy-epoch")
    plt.plot(xx, train_accs, label="train_accs")
    plt.plot(xx, valid_accs, label="valid_accs")
    plt.legend()
    plt.show()
    # save_arguments()


def read_dataset(path:str) -> ([str], [str]):
    sentences = []
    references = []
    sentence = ""
    states = ""
    train_data = open(path, encoding='utf-8')
    for line in train_data:
        line = line.strip()
        if not line:
            if not sentence:
                print("two or more null lines!!")
                continue
            sentences.append(sentence)
            references.append(states)
            sentence = ""
            states = ""
        else:
            assert len(line) == 3, "len(line) != 3"
            sentence += line[0]
            states += line[2]
    print("finish read dataset, ", len(sentences), " lines totally")
    return sentences, references


def segment(obs:str, states:str) -> str:
    segmented = ""
    assert len(obs) == len(states), "len(obs) != len(states)!"
    for i in range(0, len(obs)):
        segmented += obs[i]
        if states[i] == 'S' or states[i] == 'E':
            segmented += '/'
    return segmented


def test()->float:
    corr = 0
    total = 0
    for i in range(0, len(examples)):
        outputs = viterbi(examples[i])
        corr += sum([1 if a == b else 0 for a, b in zip(golds[i], outputs)])
        total += len(outputs)
        print("given:")
        print(segment(examples[i], golds[i]))
        print("predict:")
        print(segment(examples[i], outputs))
    return corr/total


def save_arguments():
    f = open(r"E:\源程序\PycharmProjects\lab2\unigram_template.txt", 'w')
    f.write(str(unigram_template))
    f.close()
    f = open(r"E:\源程序\PycharmProjects\lab2\\bigram_template.txt", 'w')
    f.write(str(bigram_template))
    f.close()
    f = open(r"E:\源程序\PycharmProjects\lab2\state_list.txt", 'w')
    f.write(str(state_list))
    f.close()
    f = open(r"E:\源程序\PycharmProjects\lab2\weights_dic.txt",'w')
    f.write(str(weights_dic))
    f.close()


def load_arguments():
    global unigram_template
    global bigram_template
    global state_list
    global weights_dic
    f = open(r"E:\源程序\PycharmProjects\lab2\unigram_template.txt", 'r')
    unigram_template = eval(f.read())
    f.close()
    f = open(r"E:\源程序\PycharmProjects\lab2\bigram_template.txt", 'r')
    bigram_template = eval(f.read())
    f.close()
    f = open(r"E:\源程序\PycharmProjects\lab2\state_list.txt", 'r')
    state_list = eval(f.read())
    f.close()
    f = open(r"E:\源程序\PycharmProjects\lab2\weights_dic.txt", 'r')
    weights_dic = eval(f.read())
    f.close()


if __name__ == "__main__":
    # load_arguments()
    # print("load successfully")
    init()
    train()
    test()
    save_arguments()
    print("save successfully")
    # load_arguments()
    # print("load successfully")
    # test()

