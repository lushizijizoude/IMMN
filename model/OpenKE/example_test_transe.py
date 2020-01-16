from model.OpenKE import config
from model.OpenKE import models
import tensorflow as tf
import numpy as np
import json
import os
os.environ['CUDA_VISIBLE_DEVICES']='1,2'
# (1) Set import files and OpenKE will automatically load models via tf.Saver().
con = config.Config()
# con.set_in_path("/root/python/KG/model/OpenKE/benchmarks/FB15K/")
con.set_in_path("/root/python/save/build/triples/")
con.set_test_link_prediction(True)
# con.set_test_triple_classification(True)
con.set_work_threads(8)
con.set_train_times(1000)
con.set_nbatches(100)
con.set_alpha(0.5)
con.set_margin(6.0)
con.set_bern(1)
con.set_dimension(50)
con.set_ent_neg_rate(1)
con.set_rel_neg_rate(0)
con.set_opt_method("SGD")

#Models will be exported via tf.Saver() automatically.
con.set_export_files("./res/model.vec.tf", 0)
#Model parameters will be exported to json files automatically.
con.set_out_files("./res/embedding.vec.json")
#Initialize experimental settings.
con.init()
#Set the knowledge embedding model
con.set_model(models.TransE)
#Train the model.
con.run()
#To test models after training needs "set_test_flag(True)".
con.test()