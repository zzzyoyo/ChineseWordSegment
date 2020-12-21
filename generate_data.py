import codecs

def generate_data(src: str, sen_dest: str, label_dest:str):
    fin = open(src, encoding="utf8")
    fout_sen = open(sen_dest, 'w', encoding='utf8')
    fout_label = open(label_dest, 'w', encoding='utf8')
    for line in fin:
        words = line.split()
        sentence = "".join(words)
        fout_sen.write(sentence)
        fout_sen.write('\n')
        labels = ''
        for word in words:
            if len(word) == 1:
                labels += 'S'
            else:
                labels += 'B'
                for i in range(1, len(word)-1):
                    labels += 'I'
                labels += 'E'
        fout_label.write(labels)
        fout_label.write('\n')
    fin.close()
    fout_sen.close()
    fout_label.close()


if __name__ == "__main__":
    generate_data(r"D:\Downloads\icwb2-data\my_test\RenMinData.txt_utf8", r"D:\Downloads\icwb2-data\my_test\RenMinData_input.txt",
                  r"D:\Downloads\icwb2-data\my_test\RenMinData_gold.txt")