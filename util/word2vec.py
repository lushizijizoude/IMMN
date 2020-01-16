# encoding:utf-8

import gensim
import os
import nltk
import re
import json

match_punc = re.compile(r'[^\w\s]')
import nltk

class MySentences(object):
    """
       训练词向量的语料迭代器
    """
    def __init__(self, sentence_list):
        self.sentence_list = sentence_list
        self.stop_word = nltk.corpus.stopwords.words("english")  # 使用自己的停用词表
        self.porter_stemmer = nltk.stem.PorterStemmer()
        self.lemmatizer = nltk.WordNetLemmatizer()
        self.iter_num = 0

    def __iter__(self):
        for item in self.sentence_list:
            tokens = str(item).split(" ")
            if(not tokens):
                continue
            tokens = [token.lower() for token in tokens]  # 转小写
            tokens = [token for token in tokens if not match_punc.search(token)]  # 去标点
            tokens = [token for token in tokens if not token.isdigit()]  # 去数字
            tokens = [token for token in tokens if token not in self.stop_word]  # 去停用词
            # tokens = [self.porter_stemmer.stem(token) for token in tokens] #词干提取
            # tokens = [self.lemmatizer.lemmatize(token, pos='v') for token in tokens]#词形还原
            yield tokens
        print(f"iter num is {self.iter_num}")
        self.iter_num+=1


class Word2Vec(object):
    """
    训练词向量类
    """

    def train(self, ms, model_path):
        """
        训练词向量，并且保存模型
        :return: model 词向量模型
        """
        model = gensim.models.Word2Vec(ms, min_count=2, size=100,iter = 10000,workers = 8)
        model.save(model_path,sep_limit=10 * 1024**3)
        return model

    def save_wv(self,save_path,model):
        """
        保存模型
        :return:
        """
        model.wv.save_word2vec_format(save_path+"wv", fvocab=save_path+"vocab")

    def load_wv(self,save_path):
        """
        加载词向量
        :return: w2v 词向量字典
        """
        w2v = gensim.models.KeyedVectors.load_word2vec_format(save_path+"wv",binary=False)
        return w2v

    def online_train(self, model_path, data_path):
        """
        增量训练，不会增加新的词向量，只会对原有的词向量进行修改
        :return:
        """
        model = gensim.models.Word2Vec.load(model_path)
        model.train(MySentences(data_path), total_examples=model.corpus_count, epochs=model.iter)

    def load_model(self, model_path):
        """
        获取训练的词向量及字典并将其存储到文件中
        :return:
        """
        model = gensim.models.Word2Vec.load(model_path)
        return model

    def test_vector(self, model_path, test_pair_list):
        """
        测试词向量的质量
        :param model_path:
        :param test_pair_list:
        :return:
        """
        model = gensim.models.Word2Vec.load(model_path)

        for test_pair in test_pair_list:
            pair_sim = model.similarity(test_pair[0], test_pair[1])
            print("{0}, {1}: {2}".format(test_pair[0], test_pair[1], pair_sim))


if __name__ == "__main__":

    data_path = "/home/zhangdongjie/KG_Titan/data/describe_data/"
    model_path = "/home/zhangdongjie/KG_Titan/data/embeddings/linguistic/" + "aida_word2vec_model/"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    word2vec = Word2Vec()
    with open(data_path+"sentences.json","r",encoding="utf-8") as fr:
        sentences = json.load(fr)
    ms = MySentences(sentences)
    m = word2vec.train(ms,model_path+"aida_wv_model")
    word2vec.save_wv(model_path,m)


    # m = word2vec.load_model(model_path + "aida_wv_model")
    #
    # word2vec.load_wv(model_path)
    print("ok")
    test_pair_list = []
    test_pair_list.append(("china", "beijing"))
    test_pair_list.append(("people", "dog"))
    test_pair_list.append(("water", "python"))
    test_pair_list.append(("room", "home"))


    word2vec.test_vector(model_path+"aida_wv_model", test_pair_list)
