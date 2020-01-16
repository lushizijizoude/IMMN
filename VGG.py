import numpy as np
import tensorflow as tf
from model.VGG import vgg16
from model.VGG import vgg19
from model.VGG import utils
import pickle as pkl
import os
import threading
from util.util import cosine_similarity
from util.page_rank import get_rank_value
import math
import multiprocessing
import json
import  pickle
import gc

import hashlib

def get_md5(s):
    hash = hashlib.sha1()
    s=s.encode("utf-8")
    hash.update(s)
    return hash.hexdigest()

def load_binary_file(in_file, py_version=3):
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
#VGG图像表示
class vgg_represent(object):
    max_embedding = {}
    mean_embedding = {}
    vgg_model = None
    entity_list = None
    res_max = {}
    res_mean = {}
    def __init__(self, model, data_path, save_path, entitiy_path):
        if (model == "vgg16"):
            self.vgg_model = vgg16.Vgg16()
        elif (model == "vgg19"):
            self.vgg_model = vgg19.Vgg19()
        self.model_name = "aida_" + model
        self.data_path = data_path  #google图片和wiki百科图片存放地址
        self.save_path = save_path + model + "/" #向量表示存放地址
        self.entitiy_path = entitiy_path #实体列表
        self.build()
        self.index = 0
        self.skip_num = 0
        self.sum_num = 0
        self.feature_list = []

    #获取实体列表
    def get_entity_list(self):
        if isinstance(self.entitiy_path, str):
            entity_list = []
            with open(self.entitiy_path, "r", encoding="utf-8") as fe:
                # lines = fe.read().split("\n")
                lines = json.load(fe)
                for line in lines:
                    en = line.split("\t")[0]
                    if (en):
                        entity_list.append(en)

        elif isinstance(self.entitiy_path, list):
            entity_list = self.entitiy_path
        else:
            print("entitiy_path type error!")
            return []
        self.entity_list = iter(entity_list)
        self.sum_num = len(entity_list)
        return entity_list

        # 加载embedding文件

    #获取wiki百科的图片
    def get_wiki_imgs(self, e_name):
        res = []
        try:
            img_path = self.data_path + f"entity_imgs/{e_name}/"
            if (not os.path.exists(img_path)):
                return tuple(res)
            for (root, dirs, files) in os.walk(img_path):
                for filename in files:
                    img_file = os.path.join(root, filename)
                    if (utils.filter_img_size(img_file)):
                        img = utils.load_image(img_file).reshape((224, 224, 3))
                        res.append(img)
        except:
            pass
        return tuple(res)

    # 获取wiki百科的图片
    def get_describe_imgs(self, item):
        try:
            url_md5 = get_md5(item["image"])
            img_file = "/home3/jason/KG/data/imgs/describe_imgs/" + f"{url_md5}.jpg"

            if (not os.path.exists(img_file)):
                return []
            if (utils.filter_img_size(img_file)):
                img = utils.load_image(img_file).reshape((224, 224, 3))
                return [img]
        except:
            pass
        return []

    #获取谷歌的图片
    def get_google_imgs(self, e_name):
        res = []
        try:
            img_path = self.data_path + f"google_images/{e_name}/"
            if (not os.path.exists(img_path)):
                return tuple(res)
            for (root, dirs, files) in os.walk(img_path):
                for filename in files:
                    img_file = os.path.join(root, filename)
                    if (utils.filter_img_size(img_file)):
                        # 其他过滤操作
                        img = utils.load_image(img_file).reshape((224, 224,3))
                        res.append(img)
        except:
            pass
        return tuple(res)

    #获取图片的向量表示
    def get_represent(self):
        entity_list = self.get_entity_list()
        for ename in entity_list:
            batch_wiki = self.get_wiki_imgs(ename)
            batch_google = self.get_google_imgs(ename)
            if (batch_wiki and batch_google):
                # pagerank 处理
                batch = np.concatenate((batch_wiki, batch_google), 0)
            elif (batch_wiki):
                batch = batch_wiki
            elif (batch_google):
                batch = batch_google
            else:
                self.skip_num += 1
                print(ename)
                continue
            feed_dict = {self.images: batch}
            feature = self.sess.run(self.vgg_model.fc8, feed_dict=feed_dict)
            maxfeature = np.max(feature, 0)
            self.res_max[ename] = maxfeature
            meanfeature = np.mean(feature, 0)
            self.res_mean[ename] = meanfeature
            self.index += 1
            print(f"embedding {self.index}  entity  skip {self.skip_num} entity total {len(entity_list)} entity")
        self.save_embedding()

    def prodece_worker(self, work_id):
        print(f"thread {work_id+1} start...")
        try:
            ename = next(self.entity_list)
            while (ename):
                self.index += 1
                batch_wiki = self.get_wiki_imgs(ename)
                batch_google = self.get_google_imgs(ename)
                if (batch_wiki and batch_google):
                    batch = np.concatenate((batch_wiki, batch_google), 0)
                elif (batch_wiki):
                    batch = batch_wiki
                elif (batch_google):
                    batch = batch_google
                else:
                    self.skip_num += 1
                    print(f"\tskip {ename}")
                    ename = next(self.entity_list)
                    continue
                with tf.device('/cpu:0'):
                    feed_dict = {self.images: batch}
                with tf.device('/gpu:0'):
                    feature = self.sess.run(self.vgg_model.l2_fc7, feed_dict=feed_dict)
                # gc.collect()
                # objgraph.show_growth()
                if self.index %100 == 0:
                    tf.reset_default_graph()
                fd={}
                fd["name"] = ename
                fd["feature"] = np.array(feature)
                self.feature_list.append(fd)
                print(f"worker {work_id}: embedding {self.index}  entity {ename}  skip {self.skip_num} entity total {self.sum_num} entity")
                ename = next(self.entity_list)
        except:
            print(f"worker {work_id+1} end")

        # 多线程表示学习

    def worker(self, work_id):
        print(f"thread {work_id + 1} start...")
        try:
            ename = next(self.entity_list)
            while (ename):
                batch_wiki = self.get_wiki_imgs(ename)
                batch_google = self.get_google_imgs(ename)
                if (batch_wiki and batch_google):
                    batch = np.concatenate((batch_wiki, batch_google), 0)
                elif (batch_wiki):
                    batch = batch_wiki
                elif (batch_google):
                    batch = batch_google
                else:
                    self.skip_num += 1
                    print(f"\tskip {ename}")
                    ename = next(self.entity_list)
                    continue

                feed_dict = {self.images: batch}
                feature = self.sess.run(self.vgg_model.l2_fc7, feed_dict=feed_dict)

                self.index += 1
                print(
                    f"worker {work_id}: embedding {self.index}  entity  skip {self.skip_num} entity total {self.sum_num} entity")
                ename = next(self.entity_list)
        except:
            print(f"worker {work_id + 1} end")

    def multithread_run(self, thread_num):
        self.get_entity_list()
        thread_pool = []
        for i in range(thread_num):
            t = threading.Thread(target=self.worker, args=(i,))
            thread_pool.append(t)
            t.start()
        for t in thread_pool:
            t.join()
        print("thread complete!")
        self.save_embedding()

    def multithread_process_run(self, thread_num):
        self.get_entity_list()
        thread_pool = []
        for i in range(thread_num):
            t = threading.Thread(target=self.prodece_worker, args=(i,))
            thread_pool.append(t)
            t.start()
        for t in thread_pool:
            t.join()
        # self.save_embedding()
        print("thread complete!")

    #保存图片的向量表示
    def save_embedding(self):
        if (not os.path.exists(self.save_path)):
            os.makedirs(self.save_path)
        with open(self.save_path + self.model_name + "_max.pkl", "wb") as fm:
            pkl.dump(self.res_max, fm)
        with open(self.save_path + self.model_name + "_mean.pkl", "wb") as fm:
            pkl.dump(self.res_mean, fm)

    def save_embedding_multi(self,data,save_path = None):
        if (not os.path.exists(self.save_path)):
            os.makedirs(self.save_path)
        if(not  save_path):
            save_path = self.save_path + self.model_name + "_mean.pkl"
        with open(save_path, "wb") as fm:
            pkl.dump(data, fm)
    #构建网络
    def load_embedding_multi(self,save_path = None):

        if(not  save_path):
            save_path = self.save_path + self.model_name + "_mean.pkl"
        with open(save_path, "rb") as fm:
            data=pkl.load(fm)
        print(data)

    def build(self):

        # config = tf.ConfigProto(allow_soft_placement=True)
        # config.gpu_options.allow_growth = True
        # config.gpu_options.per_process_gpu_memory_fraction = 5
        # self.sess = tf.Session(config=config)
        # config = tf.ConfigProto()
        # config.intra_op_parallelism_threads = 12
        # config.inter_op_parallelism_threads = 4
        self.sess = tf.Session()
        self.images = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.vgg_model.build(self.images)
        self.sess.graph.finalize()

def consumer_worker(data,consemer_id):
     res = []
     index = 0
     for d in data:
        index+=1
        name = d["name"]
        feature = d["feature"]
        G = {}
        for i in range(len(feature)):
            G[i] = {}
            for j in range(i + 1, len(feature)):
                cs = cosine_similarity(feature[i], feature[j])
                G[i][j] = cs
                if (j not in G):
                    G[j] = {}
                    G[j][i] = cs
        rank_frature = []
        rank_result = get_rank_value(G)
        for rr in rank_result:
            rank_frature.append(feature[rr[0]])
        rank_frature = np.array(rank_frature)

        meanfeature = np.mean(rank_frature, 0)
        mean_dic = {}
        mean_dic["name"] = name
        mean_dic["frature"] = meanfeature
        res.append(mean_dic)
        print(f"comsumer {consemer_id} complete {index} pagerank")
     return  res

def get_kg_embedding(use_pretrain_embedding =  False):
    model_name = "vgg19"
    img_path = "/home3/jason/KG/data/imgs/"
    save_path = "/home3/jason/KG/data/embeddings/vgg19_KG/"
    if (not os.path.exists(save_path)):
        os.mkdirs(save_path)
    entity_list_path = "/home3/jason/KG/data/describe_data/final_data/final_entity_set.json"
    if use_pretrain_embedding:
        if os.path.exists(save_path + model_name + "/" + model_name + "_mean.pkl"):
            emb_dic = load_binary_file(save_path + model_name + "/" + model_name + "_mean.pkl")
        else:
            emb_dic = {}
        print("use pretrain embedding")
        with open(entity_list_path, "r", encoding="utf-8") as fe:
            entity_list = json.load(fe)
            entity_list_filtered = []
            for e in entity_list:
                if e not in emb_dic:
                    entity_list_filtered.append(e)
        entity_list_path = entity_list_filtered

    process_num = 18

    rep = vgg_represent(model_name, img_path, save_path, entity_list_path)
    rep.multithread_process_run(6)
    pool = multiprocessing.Pool(processes=process_num)
    fl = rep.feature_list
    result = []

    for i in range(process_num):
        one_list = fl[math.floor(i / process_num * len(fl)):math.floor((i + 1) / process_num * len(fl))]
        result.append(pool.apply_async(consumer_worker, (one_list, i)))
    pool.close()
    pool.join()
    print("consumer complete!")
    for res in result:
        r = res.get()
        for r_dic in r:
            emb_dic[r_dic["name"]] = r_dic["frature"]
    rep.save_embedding_multi(emb_dic)
    # rep.load_embedding_multi()

def get_describe_embedding():
    model_name = "vgg19"
    save_path = "/home3/jason/KG/data/embeddings/describe/"
    if (not os.path.exists(save_path)):
        os.makedirs(save_path)
    json_file = r"/home3/jason/KG/data/describe_data/img_describe.json"
    import json
    with open(json_file, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    rep = vgg_represent(model_name, None, save_path, None)
    res_mean = {}
    for d in data_list:
        m_name = get_md5(d["image"])
        if(m_name in res_mean):
            continue
        batch = rep.get_describe_imgs(d)

        if len(batch) >0:
            feed_dict = {rep.images: batch}
            feature = rep.sess.run(rep.vgg_model.l2_fc7, feed_dict=feed_dict)
            meanfeature = np.mean(feature, 0)
            res_mean[m_name] = meanfeature
            rep.index += 1
            if(rep.index %1000 == 0 ):
                print(f"embedding {len(res_mean)} imgs")
    rep.save_embedding_multi(res_mean)

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    print("beigin")
    # get_kg_embedding(use_pretrain_embedding=True)

    get_describe_embedding()
    # with open("/root/python/save/MTKG/embedding/visual/describe/vgg19/aida_vgg19_mean.pkl", "rb") as fm:
    #     data = pkl.load(fm)
    # with open("/root/python/save/MTKG/embedding/visual/describe/vgg19/aida_vgg19_mean_filter.pkl", "rb") as fm:
    #     data1 = pkl.load(fm)
    print("ok")