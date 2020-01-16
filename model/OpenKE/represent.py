from model.OpenKE import config
from model.OpenKE import models
import pickle as pkl
import numpy as np
import os
class represent(object):
    out_path = None
    data_path = None
    entities = []
    relations = []
    def __init__(self,data_path,out_path):
        self.data_path =data_path
        self.out_path = out_path
        self.entities = self.get_entity_list()
        self.relations = self.get_relation_list()

    def get_entity_list(self):
        res = []
        file_name = self.data_path + "entity2id.txt"
        with open (file_name,"r",encoding="utf-8") as f:
            lines = f.read().split("\n")
            for line in lines:
                items = line.split("\t")
                if(len(items)!=2):
                    continue
                res.append(items[0])
        return res

    def get_relation_list(self):
        res = []
        file_name = self.data_path + "relation2id.txt"
        with open (file_name,"r",encoding="utf-8") as f:
            lines = f.read().split("\n")
            for line in lines:
                items = line.split("\t")
                if(len(items)!=2):
                    continue
                res.append(items[0])
        return res

    def get_struct_embedding(self):
        os.environ['CUDA_VISIBLE_DEVICES']='1,2'
        self.con = config.Config()
        self.con.set_in_path(self.data_path)
        self.con.set_test_link_prediction(True)
        self.con.set_test_triple_classification(True)
        self.con.set_work_threads(8)
        self.con.set_train_times(1000)
        self.con.set_nbatches(100)
        self.con.set_alpha(0.5)
        self.con.set_margin(5.0)
        self.con.set_bern(1)
        self.con.set_dimension(200)
        self.con.set_ent_neg_rate(25)
        self.con.set_rel_neg_rate(1)
        self.con.set_opt_method("SGD")


        self.con.set_export_files("./res/model.vec.tf", 0)
        # con.set_out_files(self.out_path)
        self.con.init()
        self.con.set_model(models.TransE)
        self.con.run()
        self.save_embedding()
        self.con.test()
        # self.load_embedding()

    def save_embedding(self):
        res = {}
        pp = self.con.get_parameters("ent_embeddings")
        pram = self.con.get_parameters("list")
        for i in range(len(pram["ent_embeddings"])):
            res[self.entities[i]]=np.array(pram["ent_embeddings"][i])
        for i in range(len(pram["rel_embeddings"])):
            res[self.relations[i]]=np.array(pram["rel_embeddings"][i])
        f = open(self.out_path+"struct_embedding.pkl", 'wb')
        pkl.dump(res,f)
        f.close()

    def load_embedding(self):
        f = open(self.out_path + "struct_embedding.pkl", 'rb')
        emb = pkl.load(f)
        # print(emb["邓州市"])
        print("ok")
        for item in emb.values():
            print(item)

if __name__ == "__main__":
    data_path = "/root/python/save/build/triples/seed/"
    save_path ="/root/python/save/MTKG/embedding/structure/"
    r = represent(data_path,save_path)
    r.get_struct_embedding()
    # r.load_embedding()