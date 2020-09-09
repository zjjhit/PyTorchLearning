import copy
import re


def clean_str(text):
    # text = str(text)

    # 连续多个句号替换成一个空格
    connect_list = re.findall(r"\.\.+[A-Z]", text, flags=0)
    for i in connect_list:
        second = re.findall(r"[A-Z]", i, flags=0)
        text = text.replace(i, " . " + second[0])

    connect_list = re.findall(r"\.\.+\s[A-Z]", text, flags=0)
    for i in connect_list:
        second = re.findall(r"[A-Z]", i, flags=0)
        text = text.replace(i, " . " + second[0])

    connect_list = re.findall(r"\.\.+\s[a-z0-9]", text, flags=0)
    for i in connect_list:
        second = re.findall(r"\s[a-z0-9]", i, flags=0)
        text = text.replace(i, second[0])

    connect_list = re.findall(r"\.\.+[a-z0-9]", text, flags=0)
    for i in connect_list:
        second = re.findall(r"[a-z0-9]", i, flags=0)
        text = text.replace(i, " " + second[0])

    # 标点前后插入空格
    text = text.replace("?", " ? ")
    text = text.replace(",", " , ")
    text = text.replace(".", " . ")
    text = text.replace("!", " ! ")

    # 小写单词和大写单词连一块的拆分
    connect_list = re.findall(r"\s[a-z]+[A-Z][a-z]*", text, flags=0)
    for i in connect_list:
        first = re.match(r"^[a-z]*", i[1:], flags=0)
        second = re.findall(r"[A-Z][a-z]*", i[1:], flags=0)
        text = re.sub(i, " " + first.group() + " . " + second[0], text)

    # 两个开头大写的单词连一块的拆分： MadamI'm
    connect_list = re.findall(r"\s[A-Z][a-z]+[A-Z][a-z]*", text, flags=0)
    for i in connect_list:
        first = re.match(r"^[A-Z][a-z]*", i[1:], flags=0)
        second = re.findall(r"[A-Z][a-z]*", i[1:], flags=0)
        text = re.sub(i, " " + first.group() + " . " + second[1], text)

    # 文章开头乱码去除
    pattern = r"[A-Z][a-z]+"
    pattern = re.compile(pattern)
    res = pattern.search(text)
    if res:
        text = text[res.start():]

    # 去除识别出来的噪声：Dear Sir or Madam, - I am Li Hua,
    text = re.sub(r"-", " ", text)

    # 乱码符号去除
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=\s\"\?]", "", text)

    # 多个空格替换成一个空格
    text = re.sub(r"\s+", " ", text)

    text = text.lower()
    return text


def vocab_build(texts, min_count=-float("inf")):
    """
    :param texts 二维数组
    :param min_count: 最小词频
    :return:  word2id = {'<PAD>':0, 'word1':id_1, ……， '<UNK>':id_n}
    """
    word2id_ct = {}
    for word_list in texts:
        for word in word_list.split():
            if word not in word2id_ct:
                word2id_ct[word] = [len(word2id_ct) + 1, 1]  # '词':[词序,词频]
            else:
                word2id_ct[word][1] += 1  # 词频加一

    print("len(word2id_ct):", len(word2id_ct))
    low_freq_words = []
    for word, [word_id, word_freq] in word2id_ct.items():
        if word_freq < min_count:
            low_freq_words.append(word)
    for word in low_freq_words:
        del word2id_ct[word]  # 删除低频词

    word2id = {}
    new_id = 1
    for word in word2id_ct.keys():
        word2id[word] = new_id  # word2id = {'<PAD>':0, 'word1':id_1, ......, '<UNK>':id_n}
        new_id += 1
    word2id['<UNK>'] = new_id
    word2id['<PAD>'] = 0
    print("len(word2id):", len(word2id))
    return word2id


x1_max_len = 200
x2_max_len = 200
x3_max_len = 200


###仅供参考
def word_to_seq_num(train_data_processed, word2id):
    train_seq_num = copy.deepcopy(train_data_processed)
    for index, row in train_data_processed.iterrows():
        # 分别遍历每行的两个句子，并进行分词处理
        for col_name in ['result', 'fw1', 'fw2']:
            output = []
            word_list = row[col_name].split()
            if col_name == "result":
                for i in range(x1_max_len):
                    word = word2id['<PAD>']
                    output.append(word)
                for i in range(min(x1_max_len, len(word_list))):
                    if word_list[i] not in word2id:
                        word = word2id['<UNK>']
                    else:
                        word = word2id[word_list[i]]
                    output[i] = word
            elif col_name == "fw1":
                for i in range(x2_max_len):
                    word = word2id['<PAD>']
                    output.append(word)
                for i in range(min(x2_max_len, len(word_list))):
                    if word_list[i] not in word2id:
                        word = word2id['<UNK>']
                    else:
                        word = word2id[word_list[i]]
                    output[i] = word
            else:
                for i in range(x3_max_len):
                    word = word2id['<PAD>']
                    output.append(word)
                for i in range(min(x3_max_len, len(word_list))):
                    if word_list[i] not in word2id:
                        word = word2id['<UNK>']
                    else:
                        word = word2id[word_list[i]]
                    output[i] = word
            train_seq_num.at[index, col_name] = output
    return train_seq_num
