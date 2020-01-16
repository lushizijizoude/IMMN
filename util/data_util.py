# coding: utf-8

import time
import math
import json
from datetime import timedelta
import tensorflow.contrib.keras as kr
import numpy as np
import re
import nltk
import Levenshtein
import copy
import pickle
import os
from urllib.parse import quote
from urllib.parse import unquote
PAD = "<PAD>"
match_punc = re.compile(r'[^\w\s]')
stop_word = nltk.corpus.stopwords.words("english")
import hashlib


def get_md5(s):
    hash_sha1 = hashlib.sha1()
    s = s.encode("utf-8")
    hash_sha1.update(s)
    return hash_sha1.hexdigest()


class DataUtil(object):

    def __init__(self,config):
        self.config = config
        self.data_keys = ["mention_imgs", "mention_contents", "mention_lins",\
        "pos_imgs", "pos_txts", "pos_descs",  "pos_transEs", "pos_lins" , "pos_features",\
        "neg_imgs", "neg_txts", "neg_descs",  "neg_transEs", "neg_lins", "neg_features"]

    def load_embeddings(self):
        print("load muti_struct_embedding...")
        self.muti_struct_embedding = self.load_binary_file(self.config.muti_struct_embedding_path)
        print("load transE_struct_embedding...")
        self.transE_struct_embedding = self.load_binary_file(self.config.transE_struct_embedding_path)
        print("load mention_img_embedding...")
        self.mention_img_embedding = self.load_binary_file(self.config.mention_img_embedding_path)
        print("load muti_entity_img_embedding...")
        self.muti_entity_img_embedding = self.load_binary_file(self.config.muti_entity_img_embedding_path)

        self.load_word_embedding()

    def load_word_embedding(self):
        print("load word_embedding...")
        self.words, self.word_to_id = self.read_vocab(self.config.vocab_dir + "vocab")
        self.vocab_size = len(self.words)
        self.word_embedding = (self.load_embdding(self.word_to_id, self.config.embedding_dim)
                               if self.config.is_pre_train_embed else None)

    # 加载embedding文件
    def load_binary_file(self, in_file, py_version=3):
        if not os.path.exists(in_file):
            return None
        if py_version == 2:
            with open(in_file, 'rb') as f:
                embeddings = pickle.load(f)
                return embeddings
        else:
            with open(in_file, 'rb') as f:
                u = pickle._Unpickler(f)
                u.encoding = 'latin1'
                p = u.load()
                return p


    def tocken(self, txt):
        tokens = txt.split()
        if (not tokens):
            return []
        tokens = [token.lower() for index, token in enumerate(tokens) if index < 200]  # 转小写
        tokens = [token for token in tokens if not match_punc.search(token)]  # 去标点
        tokens = [token for token in tokens if not token.isdigit()]  # 去数字
        # tokens = [token for token in tokens if token not in stop_word]  # 去停用词
        # tokens = [self.porter_stemmer.stem(token) for token in tokens] #词干提取
        # tokens = [self.lemmatizer.lemmatize(token, pos='v') for token in tokens]#词形还原
        return tokens

    def get_word_embedding(self, name):
        name = name.lower()
        name = name.replace("_", " ")
        name = name.replace("(", " ")
        name = name.replace(")", " ").strip()
        res = np.zeros((self.config.embedding_dim))
        name = name.split()
        word_num = 0
        for n in name:
            if (not n or n not in self.word_to_id):
                continue
            res += self.word_embedding[self.word_to_id[n]]
            word_num += 1
        if (word_num > 0):
            res /= word_num
        return res

    def get_time_dif(self, start_time):
        """
        获取已使用时间
        :param start_time: 起始时间
        :return:
        """
        end_time = time.time()
        time_dif = end_time - start_time
        return timedelta(seconds=int(round(time_dif)))

    def process_test_file(self, filename, word_to_id, max_length=100):

        mention_contents, mention_lins, entity_descs,  \
        entity_transE_structs, entity_lins,entity_features, group_list = self.read_test_file_json(filename)
        content_id, desc_id = [], []
        # neg
        for i in range(len(mention_contents)):
            content_id.append([word_to_id[x] for x in mention_contents[i] if x in word_to_id])
        for i in range(len(entity_descs)):
            desc_id.append([word_to_id[x] for x in entity_descs[i] if x in word_to_id])
        # 使用keras提供的pad_sequences来将文本pad为固定长度
        mention_pad = kr.preprocessing.sequence.pad_sequences(content_id, max_length, padding="post", truncating="post")
        desc_pad = kr.preprocessing.sequence.pad_sequences(desc_id, max_length, padding="post", truncating="post")
        return  mention_pad, np.array(mention_lins), desc_pad,  \
                np.array(entity_transE_structs),np.array(entity_lins), np.array(entity_features), group_list


    def read_train_file_json_itr_MNED(self, filename,section_num = 10000):
        res = {}
        for key in self.data_keys:
            res[key] = []
        group_num, group_len_sum, skip_num = 0,0,0

        print(filename)
        if filename.find("val_candidate_json.txt") >=0:
            section_num = 100000000
        with open(filename) as f:
            for line in f:
                if not line:
                    continue
                data = json.loads(line)
                mention_data = data["mention_data"]
                pos_data = data["pos_data"]
                neg_datas = data["neg_data_list"]
                mention_name = mention_data["mention_name"]
                pos_entity_name = pos_data["name"]
                pos_mention_md5 = get_md5(mention_data["img_url"])
                if pos_entity_name not in self.transE_struct_embedding:
                    skip_num += 1
                    continue
                if self.mention_img_embedding:
                    if pos_mention_md5 not in self.mention_img_embedding or \
                            pos_entity_name not in self.muti_entity_img_embedding:
                        skip_num += 1
                        continue
                    mention_img_embedding = self.mention_img_embedding[pos_mention_md5]
                    pos_img_embedding = self.muti_entity_img_embedding[pos_entity_name].reshape(-1)
                    pos_txt_embedding = self.muti_struct_embedding[pos_entity_name].reshape(-1)
                else:
                    mention_img_embedding , pos_img_embedding , pos_txt_embedding = [], [], []
                mention_content = mention_data["mention_context"]
                mention_content_tockens = self.tocken(mention_content)
                mention_name_embedding = self.get_word_embedding(mention_name)
                pos_entity_name_embedding = self.get_word_embedding(pos_entity_name)
                pos_transE = self.transE_struct_embedding[pos_entity_name].reshape(-1)
                pos_desc = pos_data["summary"]
                pos_desc_tockens = self.tocken(pos_desc)
                if not mention_content_tockens or not pos_desc_tockens or len(mention_content_tockens) <3 or len(pos_desc_tockens)<3:
                    skip_num+=1
                    continue
                #特征
                pos_feature = [value for value in pos_data["local_features"].values()]

                for neg_data in neg_datas:
                    neg_entity_name = neg_data["name"]
                    neg_entity_name_embedding = self.get_word_embedding(neg_entity_name)
                    if neg_entity_name not in self.transE_struct_embedding:
                        continue
                    neg_transE = self.transE_struct_embedding[neg_entity_name].reshape(-1)
                    if self.muti_entity_img_embedding:
                        if neg_entity_name not in self.muti_entity_img_embedding:
                            continue
                        neg_img_embedding = self.muti_entity_img_embedding[neg_entity_name].reshape(-1)
                        neg_txt_embedding = self.muti_struct_embedding[neg_entity_name].reshape(-1)
                    else:
                        neg_img_embedding ,neg_txt_embedding = [], []
                    neg_desc = neg_data["summary"]
                    neg_desc_tockens = self.tocken(neg_desc)
                    # 特征
                    neg_feature = [value for value in neg_data["local_features"].values()]

                    res["mention_contents"].append(mention_content_tockens)
                    res["mention_imgs"].append(mention_img_embedding)
                    res["mention_lins"].append(mention_name_embedding)

                    res["pos_descs"].append(pos_desc_tockens)
                    res["pos_transEs"].append(pos_transE)
                    res["pos_imgs"].append(pos_img_embedding)
                    res["pos_txts"].append(pos_txt_embedding)
                    res["pos_lins"].append(pos_entity_name_embedding)
                    res["pos_features"].append(pos_feature)

                    res["neg_descs"].append(neg_desc_tockens)
                    res["neg_transEs"].append(neg_transE)
                    res["neg_imgs"].append(neg_img_embedding)
                    res["neg_txts"].append(neg_txt_embedding)
                    res["neg_lins"].append(neg_entity_name_embedding)
                    res["neg_features"].append(neg_feature)
                    group_len_sum += 1
                group_num +=1
                if group_num % section_num ==0:
                    print(f"group num is : {group_num}")
                    print(f"group len sum is : {group_len_sum}")
                    yield res
                    res = {}
                    for key in self.data_keys:
                        res[key] = []
            print(f"skip {skip_num} data")
            print(f"group_num is {group_num}")
            print(f"group_len_sum is {group_len_sum}")
        yield res

    def process_train_file_itr_MNED(self, filename, max_length=100):
        """
        处理文件,将文件转换为id表示
        :param filename: 文件名
        :param word_to_id: word_to_id[word] = id
        :param max_length: 每个句子最大词数
        :return: 数据numpy，描述numpy， PV numpy, 标签numpy
        """
        for data in self.read_train_file_json_itr_MNED(filename):
            content_id, desc_id_neg, desc_id_pos = [], [], []
            # neg
            for i in range(len(data["mention_contents"])):
                content_id.append([self.word_to_id[x] for x in data["mention_contents"][i] if x in self.word_to_id])
            for i in range(len(data["neg_descs"])):
                desc_id_neg.append([self.word_to_id[x] for x in data["neg_descs"][i] if x in self.word_to_id])
            for i in range(len(data["pos_descs"])):
                desc_id_pos.append([self.word_to_id[x] for x in data["pos_descs"][i] if x in self.word_to_id])

            # 使用keras提供的pad_sequences来将文本pad为固定长度
            mention_pad = kr.preprocessing.sequence.pad_sequences(content_id, max_length, padding="post", truncating="post")
            desc_pad_neg = kr.preprocessing.sequence.pad_sequences(desc_id_neg, max_length, padding="post",
                                                                   truncating="post")
            desc_pad_pos = kr.preprocessing.sequence.pad_sequences(desc_id_pos, max_length, padding="post",
                                                                   truncating="post")
            data["mention_contents"] = mention_pad
            data["pos_descs"] = desc_pad_pos
            data["neg_descs"] = desc_pad_neg
            for key in self.data_keys:
                data[key] = np.array(data[key])
            yield data

    def process_test_file_itr_MNED(self, filename, max_length=100):
        """
        处理文件,将文件转换为id表示
        :param filename: 文件名
        :param word_to_id: word_to_id[word] = id
        :param max_length: 每个句子最大词数
        :return: 数据numpy，描述numpy， PV numpy, 标签numpy
        """
        data = self.read_test_file_json_itr_MNED(filename)
        content_id, desc_id_neg, desc_id_pos = [], [], []
            # neg
        for i in range(len(data["mention_contents"])):
            content_id.append([self.word_to_id[x] for x in data["mention_contents"][i] if x in self.word_to_id])
        for i in range(len(data["pos_descs"])):
            desc_id_pos.append([self.word_to_id[x] for x in data["pos_descs"][i] if x in self.word_to_id])

            # 使用keras提供的pad_sequences来将文本pad为固定长度
        mention_pad = kr.preprocessing.sequence.pad_sequences(content_id, max_length, padding="post", truncating="post")
        desc_pad_pos = kr.preprocessing.sequence.pad_sequences(desc_id_pos, max_length, padding="post",
                                                                   truncating="post")
        data["mention_contents"] = mention_pad
        data["pos_descs"] = desc_pad_pos
        for key in self.data_keys:
            data[key] = np.array(data[key])
        return data

    def read_test_file_json_itr_MNED(self, filename):
        res = {}
        for key in self.data_keys:
            res[key] = []
        res["group_list"] = []
        group_num, group_len_sum, skip_num = 0, 0, 0
        with open(filename) as f:
            for line in f:
                if not line:
                    continue
                data = json.loads(line)
                mention_data = data["mention_data"]
                pos_data = data["pos_data"]
                neg_datas = data["neg_data_list"]
                mention_name = mention_data["mention_name"]
                pos_entity_name = pos_data["name"]

                pos_mention_md5 = get_md5(mention_data["img_url"])
                if self.mention_img_embedding:
                    if pos_mention_md5 not in self.mention_img_embedding or pos_entity_name not in self.muti_entity_img_embedding:
                        skip_num += 1
                        continue
                    mention_img_embedding = self.mention_img_embedding[pos_mention_md5]
                    pos_img_embedding = self.muti_entity_img_embedding[pos_entity_name].reshape(-1)
                    pos_txt_embedding = self.muti_struct_embedding[pos_entity_name].reshape(-1)
                else:
                    mention_img_embedding = []
                    pos_img_embedding = []
                    pos_txt_embedding = []

                mention_content = mention_data["mention_context"]
                mention_content_tockens = self.tocken(mention_content)
                mention_name_embedding = self.get_word_embedding(mention_name)
                pos_entity_name_embedding = self.get_word_embedding(pos_entity_name)
                if pos_entity_name not in self.transE_struct_embedding:
                    continue
                pos_transE = self.transE_struct_embedding[pos_entity_name].reshape(-1)
                pos_desc = pos_data["summary"]
                pos_desc_tockens = self.tocken(pos_desc)

                pos_feature = [value for value in pos_data["local_features"].values()]
                if not mention_content_tockens or not pos_desc_tockens or len(mention_content_tockens) < 3 or len(
                        pos_desc_tockens) < 3:
                    continue
                else:
                    res["mention_contents"].append(mention_content_tockens)
                    res["mention_imgs"].append(mention_img_embedding)
                    res["mention_lins"].append(mention_name_embedding)

                    res["pos_descs"].append(pos_desc_tockens)
                    res["pos_transEs"].append(pos_transE)
                    res["pos_imgs"].append(pos_img_embedding)
                    res["pos_txts"].append(pos_txt_embedding)
                    res["pos_lins"].append(pos_entity_name_embedding)
                    res["pos_features"].append(pos_feature)

                group_len = 1
                for neg_data in neg_datas:
                    neg_entity_name = neg_data["name"]
                    neg_entity_name_embedding = self.get_word_embedding(neg_entity_name)
                    if neg_entity_name not in self.transE_struct_embedding:
                        continue
                    neg_transE = self.transE_struct_embedding[neg_entity_name].reshape(-1)
                    if self.muti_entity_img_embedding:
                        if (neg_entity_name not in self.muti_entity_img_embedding):
                            continue
                        neg_img_embedding = self.muti_entity_img_embedding[neg_entity_name].reshape(-1)
                        neg_txt_embedding = self.muti_struct_embedding[neg_entity_name].reshape(-1)
                    else:
                        neg_img_embedding = []
                        neg_txt_embedding = []
                    neg_desc = neg_data["summary"]
                    neg_desc_tockens = self.tocken(neg_desc)

                    # 特征
                    neg_feature = [value for value in neg_data["local_features"].values()]

                    res["mention_contents"].append(mention_content_tockens)
                    res["mention_imgs"].append(mention_img_embedding)
                    res["mention_lins"].append(mention_name_embedding)

                    res["pos_descs"].append(neg_desc_tockens)
                    res["pos_transEs"].append(neg_transE)
                    res["pos_imgs"].append(neg_img_embedding)
                    res["pos_txts"].append(neg_txt_embedding)
                    res["pos_lins"].append(neg_entity_name_embedding)
                    res["pos_features"].append(neg_feature)
                    group_len += 1
                    group_len_sum += 1
                group_num += 1
                res["group_list"].append(group_len)
        tmp_group_list = [item for item in res["group_list"] if item >=5]
        print(f"len group_list is {len(tmp_group_list)}")
        print(f"group sum is {sum(tmp_group_list)}")
        return res

    def build_vocab(self, data_dir_list, vocab_dir):
        """
        根据训练集构建词汇表，存储
        :param data_dir_list: 数据表路径list
        :param vocab_dir: 词汇表存储路径
        :param vocab_size:
        :return:
        """
        all_data = []
        for dir_item in data_dir_list:
            contents, descs, _ = self.read_file(dir_item)

            for content_item in contents:
                all_data.extend(content_item)

            for desc_item in descs:
                all_data.extend(desc_item)

        all_data_set = set(all_data)
        # 添加一个 <PAD> 来将所有文本pad为同一长度
        all_data_set.add(PAD)
        open(vocab_dir, mode='w').write('\n'.join(all_data_set).encode("utf-8"))

    def read_vocab(self, vocab_dir):
        """
        读取词汇表,将word转化为id表示
        :param vocab_dir: 词汇表路径
        :return: words列表, word_to_id[word] = word_id
        """
        words = ["_padding"]
        with open(vocab_dir) as fp:
            vacb_lines = fp.read().split("\n")
            for vline in vacb_lines:
                word_ = vline.split()
                if (word_):
                    words.append(word_[0])
        word_to_id = dict(zip(words, range(len(words))))

        # 防止词汇表部分词无法充当key,导致max(word_to_id.values) != len(word_to_id) - 1
        word_to_id = dict(zip(word_to_id.keys(), range(len(word_to_id))))
        return words, word_to_id

    def load_embdding(self, word_to_id, dim):
        """
        加载预训练词向量
        :param embed_file_path:
        :param word_to_id:
        :param dim:
        :return:  embeddings np
        """
        from util import word2vec
        embeddings = np.zeros([len(word_to_id), dim])
        wv = word2vec.Word2Vec().load_model(self.config.pre_train_embed_path).wv
        for we in word_to_id:
            if we and (we in wv):
                embeddings[word_to_id[we]] = np.asarray(np.asarray(wv[we]))
        return embeddings

    def batch_iter(self, data, batch_size=64,
                   is_random=False):
        if (not data):
            return
        data_len = len(data[0])
        # 批次数
        num_batch = int((data_len - 1) / batch_size) + 1
        data_shuffle = copy.deepcopy(data)
        # 对每组数据构建正负例对加入到新的list之中
        # 将数据打乱
        if is_random:
            indices = np.random.permutation(np.arange(data_len))
        else:
            indices = np.arange(data_len)
            # np.random.shuffle(data_shuffle[i])
        for i in range(num_batch):
            start_id = i * batch_size
            end_id = min((i + 1) * batch_size, data_len)
            batch_truple = []
            for item in data_shuffle:
                batch_truple.append(item[indices[start_id:end_id]])
            yield tuple(batch_truple)

    def section_batch_iter(self, save_path, batch_size=64,
                           is_random=False):

        for (root, dirs, files) in os.walk(save_path):
            for filename in files:
                with open(save_path + filename, "rb") as f:
                    data = pickle.load(f)
                if (not data):
                    return
                data_len = len(data[0])
                # 批次数
                num_batch = int((data_len - 1) / batch_size) + 1
                data_shuffle = copy.deepcopy(data)
                # 对每组数据构建正负例对加入到新的list之中
                # 将数据打乱
                if is_random:
                    indices = np.random.permutation(np.arange(data_len))
                else:
                    indices = np.arange(data_len)

                    # np.random.shuffle(data_shuffle[i])
                for i in range(num_batch):
                    start_id = i * batch_size
                    end_id = min((i + 1) * batch_size, data_len)
                    batch_truple = []
                    for item in data_shuffle:
                        batch_truple.append(item[indices[start_id:end_id]])
                    yield tuple(batch_truple)

    # 数据分块存储
    def section_save(self, data, seg_num, save_path,index = -1):
        data_len = len(data[0])
        segs = data_len // seg_num
        if (data_len % seg_num != 0):
            segs += 1
        is_random = True
        if is_random:
            indices = np.random.permutation(np.arange(data_len))
        for i in range(len(data)):
            data[i] = data[i][indices]
        for i in range(segs):
            tmp = []
            begin = i * seg_num
            end = min((i + 1) * seg_num, data_len)
            if index >=0:
                with open(save_path + f"{index}_{i}_seg.pkl", "wb") as f:
                    for item in data:
                        tmp.append(item[begin:end])
                    pickle.dump(tmp, f)
            else:
                with open(save_path + f"{i}_seg.pkl", "wb") as f:
                    for item in data:
                        tmp.append(item[begin:end])
                    pickle.dump(tmp, f)


if __name__ == "__main__":
    # print("ok")
    from util.config import Config_Util
    config = Config_Util()
    d = DataUtil(config)
    d.data_expend("/home3/jason/KG/data/describe_data/final_data/")
    print("ok")
    exit()
    d.load_embeddings()
    d.read_train_file_json_itr("/home3/jason/KG/data/describe_data/wikilink_format_final/train_candidate_expend_local_json_merge.txt")
    d.read_test_file_json("/home3/jason/KG/data/describe_data/wikilink_format_final/test_candidate_expend_local_json_merge.txt")
    # d.data_expend("/home3/jason/KG/data/describe_data/final_data/")
    print("data_expend ok")
    exit()
