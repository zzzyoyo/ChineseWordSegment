import math

TRAIN_DATA1 = "E:\大三上\智能系统\LAB2\dataset\dataset1\\train.utf8"
TRAIN_DATA2 = "E:\大三上\智能系统\LAB2\dataset\dataset2\\train.utf8"
LABELS = "E:\大三上\智能系统\LAB2\dataset\dataset1\labels.utf8"

state_list = []
Pi_dic = {}  # initial state probabilities
A_dic = {}  # state transition probabilities
B_dic = {}  # symbol emission probabilities
Min_dic = {}  # emission probabilities for unknown symbol
Prev_Count_dic = {}  # counts of transitions from every state i
Prev_Curr_dic = {}  # counts of transitions from every state i to j


def init():
    label_file = open(LABELS)
    for state in label_file:
        state_list.append(state.strip())
    print(state_list)

    for state in state_list:
        A_dic[state] = {}
        B_dic[state] = {}
        Min_dic[state] = {}
        Prev_Curr_dic[state] = {}
        Pi_dic[state] = 0.0
        Prev_Count_dic[state] = 0
        for state1 in state_list:
            A_dic[state][state1] = 0.0
            B_dic[state][state1] = {}
            Min_dic[state][state1] = 0
            Prev_Curr_dic[state][state1] = 0


def test_sum():
    print("---B_dic---")
    for key in B_dic:
        for key1 in B_dic[key]:
            sum_of_p = 0.0
            for key2 in B_dic[key][key1]:
                sum_of_p += B_dic[key][key1][key2]
            print(sum_of_p)
    print("---A_dic---")
    for key in A_dic:
        sum_of_p = 0.0
        for key1 in A_dic[key]:
            sum_of_p += A_dic[key][key1]
        print(sum_of_p)
    print("---Pi_dic---")
    sum_of_p = 0.0
    for key in Pi_dic:
        sum_of_p += Pi_dic[key]
    print(sum_of_p)


def train(log: bool):
    train_data = open(TRAIN_DATA1, encoding='utf-8').readlines()
    # 第二个数据集也读进来
    train_data[len(train_data):len(train_data)] = open(TRAIN_DATA2, encoding='utf-8').readlines()
    init()
    line_num = 0
    last_state = None  # last state of current position, None if it's in start of a line
    last_obs = None  # last observation of current position
    for pair in train_data:
        pair = pair.strip()
        if not pair:
            # a new line
            line_num += 1
            last_state = None
            last_obs = None
            continue
        pair = pair.split()
        # transition
        if not last_state:
            # start of a line
            Pi_dic[pair[1]] += 1
            last_state = pair[1]
            last_obs = pair[0]
        else:
            Prev_Curr_dic[last_state][pair[1]] += 1
            Prev_Count_dic[last_state] += 1
            # emission
            if last_obs in B_dic[last_state][pair[1]]:
                B_dic[last_state][pair[1]][last_obs] += 1
            else:
                B_dic[last_state][pair[1]][last_obs] = 1
            last_state = pair[1]
            last_obs = pair[0]
    line_num += 1  # 最后一句没有一个空行作为结尾，因此还要加一才是句子的数量
    print("totally ", line_num, " lines.")

    # 归一化
    for key in Pi_dic:
        Pi_dic[key] = Pi_dic[key] / line_num
    for key in A_dic:
        for key1 in A_dic[key]:
            A_dic[key][key1] = Prev_Curr_dic[key][key1] / Prev_Count_dic[key]
    for key in B_dic:
        for key1 in B_dic[key]:
            for word in B_dic[key][key1]:
                B_dic[key][key1][word] /= Prev_Curr_dic[key][key1]

    # 取对数
    if log:
        change_to_log()

    # compute min probabilities
    for key in Min_dic:
        for key1 in Min_dic[key]:
            probs = B_dic[key][key1].values()  # probs可能不存在，因为B后面不可能是S
            Min_dic[key][key1] = min(probs if probs else [0])
    # print(Pi_dic)
    # print(B_dic)
    # print(A_dic)


def viterbi(obs:str) -> (float, str):
    V = [{}]
    Path = []
    for key in state_list:
        V[0][key] = Pi_dic[key]
    for t in range(1, len(obs) + 1):
        V.append({})
        Path.append({})
        for key in state_list:
            (prob, last_state) = max([(V[t-1][y0] * A_dic[y0][key] * B_dic[y0][key].get(obs[t-1], Min_dic[y0][key]), y0) for y0 in state_list if V[t - 1][y0] > 0])
            V[t][key] = prob
            Path[t-1][key] = last_state  # 令state(t)为key的概率最大的state(t-1)
    (prob, state) = max([(V[len(obs)][y], y) for y in state_list])
    state_str = ""
    for t in range(len(obs)-1, -1, -1):
        state_str = Path[t][state] + state_str
        state = Path[t][state]  # 令当前状态的概率最大的上一个状态
    return prob, state_str


def viterbi_sum(obs:str) -> (float, str):
    V = [{}]
    Path = []
    for key in state_list:
        V[0][key] = Pi_dic[key]
    for t in range(1, len(obs) + 1):
        V.append({})
        Path.append({})
        for key in state_list:
            (prob, last_state) = max([(V[t-1][y0] + A_dic[y0][key] + B_dic[y0][key].get(obs[t-1], Min_dic[y0][key]), y0) for y0 in state_list ])
            V[t][key] = prob
            Path[t-1][key] = last_state  # 令state(t)为key的概率最大的state(t-1)
    (prob, state) = max([(V[len(obs)][y], y) for y in state_list])
    state_str = ""
    for t in range(len(obs)-1, -1, -1):
        state_str = Path[t][state] + state_str
        state = Path[t][state]  # 令当前状态的概率最大的上一个状态
    return prob, state_str


def predict(obs: str, log: bool) -> str:
    if log:
        return viterbi_sum(obs)[1]
    else:
        sentences = obs.split('，')
        states = ""
        for sent in sentences:
            states += viterbi(sent)[1]
            states += 'S'
        return states[:-1]


def segment(obs:str, states:str) -> str:
    segmented = ""
    assert len(obs) == len(states), "len(obs) != len(states)!"
    for i in range(0, len(obs)):
        segmented += obs[i]
        if states[i] == 'S' or states[i] == 'E':
            segmented += '/'
    return segmented


def change_to_log():
    for key in B_dic:
        for key1 in B_dic[key]:
            for word in B_dic[key][key1]:
                B_dic[key][key1][word] = math.log(B_dic[key][key1][word]) if B_dic[key][key1][word] > 0 else float(-2 ** 31)

    for key in A_dic:
        for key1 in A_dic[key]:
            A_dic[key][key1] = math.log(A_dic[key][key1]) if A_dic[key][key1] > 0 else float(-2 ** 31)
    for key in Pi_dic:
        Pi_dic[key] = math.log(Pi_dic[key]) if Pi_dic[key] > 0 else float(-2 ** 31)


def test():
    examples = open("E:\大三上\智能系统\LAB2\lab2_submission\example_dataset\input.utf8", encoding="utf8").readlines()
    examples = [ele.strip() for ele in examples]
    golds = open("E:\大三上\智能系统\LAB2\lab2_submission\example_dataset\gold.utf8", encoding="utf8").readlines()
    golds = [ele.strip() for ele in golds]
    for i in range(0, len(examples)):
        output = predict(examples[i], True)
        print("given:")
        print(segment(examples[i], golds[i]))
        print("predict:")
        print(segment(examples[i], output))


if __name__ == "__main__":
    # HMM训练速度很快，可以当场train
    train(True)
    test()
