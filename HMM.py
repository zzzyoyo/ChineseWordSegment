from typing import List

TRAIN_DATA = "E:\大三上\智能系统\LAB2\dataset\dataset1\\train.utf8"
LABELS = "E:\大三上\智能系统\LAB2\dataset\dataset1\labels.utf8"

state_list = []
Pi_dic = {}  # initial state probabilities
A_dic = {}  # state transition probabilities
B_dic = {}  # symbol emission probabilities
Average_dic = {}  # average emission probabilities for unknown symbol
Count_dic = {}  # counts of every state
Prev_Count_dic = {}  # counts of every state except the last one


def init():
    label_file = open(LABELS)
    for state in label_file:
        state_list.append(state.strip())
    print(state_list)

    for state in state_list:
        A_dic[state] = {}
        for state1 in state_list:
            A_dic[state][state1] = 0.0
    for state in state_list:
        Pi_dic[state] = 0.0
        B_dic[state] = {}
        Count_dic[state] = 0
        Prev_Count_dic[state] = 0
        Average_dic[state] = 0


def test_sum():
    print("---B_dic---")
    for key in B_dic:
        sum_of_p = 0.0
        for key1 in B_dic[key]:
            sum_of_p += B_dic[key][key1]
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



def train():
    train_data = open(TRAIN_DATA, encoding='utf-8')
    init()
    line_num = 0
    last_state = None  # last state of current position, None if it's in start of a line
    for pair in train_data:
        pair = pair.strip()
        if not pair:
            # a new line
            line_num += 1
            last_state = None
            continue
        pair = pair.split()
        Count_dic[pair[1]] += 1
        # emission
        if pair[0] in B_dic[pair[1]]:
            B_dic[pair[1]][pair[0]] += 1
        else:
            B_dic[pair[1]][pair[0]] = 1
        # transition
        if not last_state:
            # start of a line
            Pi_dic[pair[1]] += 1
            last_state = pair[1]
        else:
            A_dic[last_state][pair[1]] += 1
            Prev_Count_dic[last_state] += 1
            last_state = pair[1]
    line_num += 1  # 最后一句没有一个空行作为结尾，因此还要加一才是句子的数量
    print("totally ",line_num, " lines.")

    # 归一化
    for key in Pi_dic:
        Pi_dic[key] = Pi_dic[key] / line_num
    for key in A_dic:
        for key1 in A_dic[key]:
            A_dic[key][key1] /= Prev_Count_dic[key]
    for key in B_dic:
        for word in B_dic[key]:
            B_dic[key][word] /= Count_dic[key]

    # compute average probabilities
    for key in Average_dic:
        Average_dic[key] = 1.0 / len(B_dic[key])
    # print(Pi_dic)
    # print(B_dic)
    # print(A_dic)


def viterbi(obs: str) -> (float, str):
    V = [{}]
    path = {}
    for y in state_list:   #初始值
        V[0][y] = Pi_dic[y] * B_dic[y].get(obs[0], Average_dic[y])   # max P(o0,o1,o2...ot,Xt=y), 用get而不是用[]是因为又可能key不存在，此时返回默认值，不能返回0，否则一遇到一个不认识的后面就全部是0了
        path[y] = [y]
    for t in range(1,len(obs)):
        if t == 110:
            print(V[t-1])
            for y in state_list:
                print(V[t-1]['B'],A_dic['B'][y],B_dic[y].get(obs[t], Average_dic[y]))
                x = V[t-1]['B'] * A_dic['B'][y] * B_dic[y].get(obs[t], Average_dic[y])
                print(x)
        V.append({})
        newpath = {}
        for y in state_list:      #从y0 -> y状态的递归
            (prob, state) = max([(V[t-1][y0] * A_dic[y0][y] * B_dic[y].get(obs[t], Average_dic[y]), y0) for y0 in state_list if V[t - 1][y0] > 0])
            V[t][y] =prob
            newpath[y] = path[state] + [y]
        path = newpath  #记录状态序列
    (prob, state) = max([(V[len(obs) - 1][y], y) for y in state_list])  #在最后一个位置，以y状态为末尾的状态序列的最大概率
    return prob, "".join(path[state])  # 返回概率和状态序列


def predict(obs:str)->str:
    sentences = obs.split('，')
    states = ""
    for sent in sentences:
        states += viterbi(sent)[1]
        states += 'S'
    return states[:-1]


if __name__ == "__main__":
    train()
    examples = open("E:\大三上\智能系统\LAB2\lab2_submission\example_dataset/input.utf8", encoding="utf8").readlines()
    examples = [ele.strip() for ele in examples]
    # outputs = [viterbi(ele)[1] for ele in examples]
    # outputs = []
    # for ele in examples:
    #     print(ele)
    #     output = viterbi(ele)[1]
    #     print(output)
    #     outputs.append(output)
    outputs = predict(examples[1])
    print(outputs)
