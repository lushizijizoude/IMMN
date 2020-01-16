# coding:utf-8
import os
import numpy as np
import tensorflow as tf
from model.MTRL import util as u
import operator
import threading
from multiprocessing import Pool
import multiprocessing
# from pathos.multiprocessing import  Pool
from multiprocessing.dummy import Pool as ThreadPool
import time


class mtrl(object):
    logs_path = "log"

    def __init__(self):
        self.model_config()
        self.load_data()
        self.build()
        # self.train_multiprocess()
        self.train()
        self.save_multi_entity_embedding()
        self.test()

    def model_config(self):
        self.base_dir = "/home3/jason/KG/data/"
        self.thread_num = 8
        self.process_num = 8

        self.relation_structural_embeddings_size = 100
        self.entity_structural_embeddings_size = 100
        self.entity_multimodal_embeddings_size = 4096
        self.mapping_size = 100
        self.data_dir = "/home3/jason/KG/data/KG_data/IMMN_data/all_relations/"
        self.dropout_ratio = 0.2
        self.model_id = "MNED"

        self.entity_structural_embeddings_file = self.base_dir + "embeddings/struct/aida_tranE_100_SGDstruct_embedding.pkl"
        self.entity_multimodal_embeddings_file = self.base_dir + "/embeddings/vgg19_KG/vgg19/aida_vgg19_mean.pkl"
        self.relation_structural_embeddings_file = self.entity_structural_embeddings_file

        self.dropout_ratio = 0.0
        self.margin = 10
        self.training_epochs = 1000
        self.batch_size = 100
        self.batch_size_valid = 1024
        self.display_step = 1

        self.activation_function = tf.nn.tanh
        self.initial_learning_rate = 0.001
        self.all_triples_file = self.data_dir + "all.txt"  # "
        self.train_triples_file = self.data_dir + "train.txt"  #
        self.test_triples_file = self.data_dir + "test.txt"
        self.valid_triples_file = self.data_dir + "val.txt"

        self.checkpoint_best_valid_dir = self.base_dir + "weights/best_" + self.model_id + "/"
        self.checkpoint_current_dir = self.base_dir + "weights/current_" + self.model_id + "/"
        self.results_dir = self.base_dir + "results/results_" + self.model_id + "/"
        self.producer_train_dir = "/dev/tmp/producer/train_" + self.model_id + "/"


        if not os.path.exists(self.checkpoint_best_valid_dir):
            os.makedirs(self.checkpoint_best_valid_dir)

        if not os.path.exists(self.checkpoint_current_dir):
            os.makedirs(self.checkpoint_current_dir)

        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        self.model_current_weights_file = self.checkpoint_current_dir + self.model_id + "_current"
        self.current_model_meta_file = self.checkpoint_current_dir + self.model_id + "_current.meta"

        self.model_weights_best_valid_file = self.checkpoint_best_valid_dir + self.model_id + "_best_hits"
        self.best_valid_model_meta_file = self.checkpoint_best_valid_dir + self.model_id + "_best_hits.meta"

        self.result_file = self.results_dir + self.model_id + "_results.txt"
        self.log_file = self.results_dir + self.model_id + "_log.txt"

        self.embedding_save_path = self.base_dir + f"embeddings/MTRL/"
        if not os.path.exists(self.embedding_save_path):
            os.makedirs(self.embedding_save_path)

    def filter_triples(self, triples):
        res = []
        for t in triples:
            if t[0] in self.entity_embeddings_txt and t[0] in self.entity_embeddings_img \
                    and t[1] in self.entity_embeddings_txt and t[1] in self.entity_embeddings_img \
                    and t[2] in self.entity_embeddings_txt:
                res.append(t)
        return res

    def filter_entity(self, entity_list):
        res = []
        for e in entity_list:
            if e in self.entity_embeddings_img and e in self.entity_embeddings_txt:
                res.append(e)
        return res

    def filter_relation(self, relation_list):
        res = []
        for e in relation_list:
            if e in self.entity_embeddings_txt:
                res.append(e)
        return res

    def predict_best_tail(self, test_triple, full_triple_list, full_entity_list, entity_embeddings_txt,
                          entity_embeddings_img,
                          full_relation_embeddings):
        results = {}
        gt_head = test_triple[0]
        gt_head_embeddings_txt = entity_embeddings_txt[gt_head]
        gt_head_embeddings_img = entity_embeddings_img[gt_head]

        gt_rel = test_triple[2]
        gt_relation_embeddings = full_relation_embeddings[gt_rel]
        gt_tail_org = test_triple[1]
        gt_tail = u.get_correct_tails(gt_head, gt_rel, full_triple_list)

        total_batches = len(full_entity_list) // self.batch_size + 1

        predictions = []
        for batch_i in range(total_batches):
            start = self.batch_size * (batch_i)
            end = self.batch_size * (batch_i + 1)

            tails_embeddings_list_txt = []
            tails_embeddings_list_img = []

            head_embeddings_list_txt = np.tile(gt_head_embeddings_txt, (len(full_entity_list[start:end]), 1))
            head_embeddings_list_img = np.tile(gt_head_embeddings_img, (len(full_entity_list[start:end]), 1))
            full_relation_embeddings = np.tile(gt_relation_embeddings, (len(full_entity_list[start:end]), 1))

            for i in range(len(full_entity_list[start:end])):
                tails_embeddings_list_txt.append(entity_embeddings_txt[full_entity_list[start + i]])
                tails_embeddings_list_img.append(entity_embeddings_img[full_entity_list[start + i]])

            sub_predictions = self.predict_tail(head_embeddings_list_txt, head_embeddings_list_img,
                                                full_relation_embeddings,
                                                tails_embeddings_list_txt, tails_embeddings_list_img)
            for p in sub_predictions:
                predictions.append(p)

        predictions = [predictions]
        for i in range(0, len(predictions[0])):
            if full_entity_list[i] == gt_head and gt_head not in gt_tail:
                pass
                # results[full_entity_list[i]] = 0
            else:
                results[full_entity_list[i]] = predictions[0][i]

        sorted_x = sorted(results.items(), key=operator.itemgetter(1), reverse=False)
        top_10_predictions = [x[0] for x in sorted_x[:10]]
        sorted_keys = [x[0] for x in sorted_x]
        index_correct_tail_raw = sorted_keys.index(gt_tail_org)

        gt_tail_to_filter = [x for x in gt_tail if x != gt_tail_org]
        for key in gt_tail_to_filter:
            if key in results:
                del results[key]

        sorted_x = sorted(results.items(), key=operator.itemgetter(1), reverse=False)
        sorted_keys = [x[0] for x in sorted_x]
        index_tail_head_filter = sorted_keys.index(gt_tail_org)

        return (index_correct_tail_raw + 1), (index_tail_head_filter + 1), top_10_predictions


    def predict_tail(self, head_embedding_txt, head_embedding_img, relation_embedding, tails_embedding_txt,
                     tails_embeddings_img):

        r_input = self.graph.get_tensor_by_name("input/r_input:0")
        h_pos_txt_input = self.graph.get_tensor_by_name("input/h_pos_txt_input:0")
        t_pos_txt_input = self.graph.get_tensor_by_name("input/t_pos_txt_input:0")

        h_pos_img_input = self.graph.get_tensor_by_name("input/h_pos_img_input:0")
        t_pos_img_input = self.graph.get_tensor_by_name("input/t_pos_img_input:0")

        keep_prob = self.graph.get_tensor_by_name("input/keep_prob:0")

        h_r_t_pos = self.graph.get_tensor_by_name("cosine/h_r_t_pos:0")

        predictions = h_r_t_pos.eval(feed_dict={r_input: relation_embedding,
                                                h_pos_txt_input: np.asarray(head_embedding_txt),
                                                t_pos_txt_input: np.asarray(tails_embedding_txt),
                                                h_pos_img_input: np.asarray(head_embedding_img),
                                                t_pos_img_input: np.asarray(tails_embeddings_img),
                                                keep_prob: 1.0})
        return predictions

    def predict_head(self, tail_embeddings_list_txt, tail_embeddings_list_img, full_relation_embeddings,
                     heads_embeddings_list_txt, heads_embeddings_list_img):

        r_input = self.graph.get_tensor_by_name("input/r_input:0")
        h_pos_txt_input = self.graph.get_tensor_by_name("input/h_pos_txt_input:0")
        t_pos_txt_input = self.graph.get_tensor_by_name("input/t_pos_txt_input:0")

        h_pos_img_input = self.graph.get_tensor_by_name("input/h_pos_img_input:0")
        t_pos_img_input = self.graph.get_tensor_by_name("input/t_pos_img_input:0")

        keep_prob = self.graph.get_tensor_by_name("input/keep_prob:0")

        t_r_h_pos = self.graph.get_tensor_by_name("cosine/t_r_h_pos:0")

        predictions = t_r_h_pos.eval(feed_dict={r_input: full_relation_embeddings,
                                                h_pos_txt_input: np.asarray(heads_embeddings_list_txt),
                                                t_pos_txt_input: np.asarray(tail_embeddings_list_txt),
                                                h_pos_img_input: np.asarray(heads_embeddings_list_img),
                                                t_pos_img_input: np.asarray(tail_embeddings_list_img),
                                                keep_prob: 1.0})

        return predictions

    def predict_best_head(self, test_triple, full_triple_list, full_entity_list, entity_embeddings_txt,
                          entity_embeddings_img,
                          full_relation_embeddings):

        results = {}
        gt_tail = test_triple[1]  # tail
        gt_tail_embeddings_txt = entity_embeddings_txt[gt_tail]  # tail embeddings
        gt_tail_embeddings_img = entity_embeddings_img[gt_tail]

        gt_rel = test_triple[2]
        gt_relation_embeddings = full_relation_embeddings[gt_rel]

        gt_head_org = test_triple[0]
        gt_head = u.get_correct_heads(gt_tail, gt_rel, full_triple_list)

        total_batches = len(full_entity_list) // self.batch_size + 1

        predictions = []
        for batch_i in range(total_batches):
            start = self.batch_size * (batch_i)
            end = self.batch_size * (batch_i + 1)
            heads_embeddings_list_txt = []
            heads_embeddings_list_img = []

            tail_embeddings_list_txt = np.tile(gt_tail_embeddings_txt, (len(full_entity_list[start:end]), 1))
            tail_embeddings_list_img = np.tile(gt_tail_embeddings_img, (len(full_entity_list[start:end]), 1))
            full_relation_embeddings = np.tile(gt_relation_embeddings, (len(full_entity_list[start:end]), 1))

            for i in range(len(full_entity_list[start:end])):
                heads_embeddings_list_txt.append(entity_embeddings_txt[full_entity_list[start + i]])
                heads_embeddings_list_img.append(entity_embeddings_img[full_entity_list[start + i]])

            sub_predictions = self.predict_head(tail_embeddings_list_txt, tail_embeddings_list_img,
                                                full_relation_embeddings,
                                                heads_embeddings_list_txt, heads_embeddings_list_img)

            for p in sub_predictions:
                predictions.append(p)

        predictions = [predictions]

        for i in range(0, len(predictions[0])):
            if full_entity_list[i] == gt_tail and gt_tail not in gt_head:

                #    #results[full_entity_list[i]] = 0
                pass
            else:
                results[full_entity_list[i]] = predictions[0][i]

        sorted_x = sorted(results.items(), key=operator.itemgetter(1), reverse=False)
        top_10_predictions = [x[0] for x in sorted_x[:10]]
        sorted_keys = [x[0] for x in sorted_x]
        index_correct_head_raw = sorted_keys.index(gt_head_org)

        gt_tail_to_filter = [x for x in gt_head if x != gt_head_org]
        # remove the correct tails from the predictions
        for key in gt_tail_to_filter:
            if(key in results):
                del results[key]

        sorted_x = sorted(results.items(), key=operator.itemgetter(1), reverse=False)
        sorted_keys = [x[0] for x in sorted_x]
        index_head_filter = sorted_keys.index(gt_head_org)

        return (index_correct_head_raw + 1), (index_head_filter + 1), top_10_predictions

    def max_norm_regulizer(self, threshold=1.0, axes=1, name="max_norm", collection="max_norm"):
        def max_norm(weights):
            clipped = tf.clip_by_norm(weights, clip_norm=threshold, axes=axes)
            clip_weights = tf.assign(weights, clipped, name=name)
            tf.add_to_collection(collection, clip_weights)
            return None

        return max_norm

    def load_data(self):
        self.relation_embeddings = u.load_binary_file(self.relation_structural_embeddings_file)
        self.entity_embeddings_txt = u.load_binary_file(self.entity_structural_embeddings_file)
        self.entity_embeddings_img = u.load_binary_file(self.entity_multimodal_embeddings_file)
        # Remove triples that don't have embeddings
        self.all_train_test_valid_triples, self.entity_list = u.load_training_triples(self.all_triples_file)
        self.entity_list = self.filter_entity(self.entity_list)
        self.all_train_test_valid_triples = self.filter_triples(self.all_train_test_valid_triples)


        self.triples_set = [t[0] + "_" + t[1] + "_" + t[2] for t in self.all_train_test_valid_triples]
        self.triples_set = set(self.triples_set)
        print("#entities", len(self.entity_list), "#total triples", len(self.all_train_test_valid_triples))
        self.training_data = u.load_freebase_triple_data_multimodal(self.train_triples_file, self.entity_embeddings_txt,
                                                                    self.entity_embeddings_img,
                                                                    self.relation_embeddings)
        print("#training data", len(self.training_data))
        self.valid_data = u.load_freebase_triple_data_multimodal(self.valid_triples_file, self.entity_embeddings_txt,
                                                                 self.entity_embeddings_img, self.relation_embeddings)
        self.h_data_valid_txt, self.h_data_valid_img, self.r_data_valid, self.t_data_valid_txt, \
        self.t_data_valid_img, self.t_neg_data_valid_txt, self.t_neg_data_valid_img, \
        self.h_neg_data_valid_txt, self.h_neg_data_valid_img = \
            u.get_batch_with_neg_heads_and_neg_tails_multimodal(self.valid_data, self.triples_set,
                                                                self.entity_list, 0,
                                                                len(self.valid_data),
                                                                self.entity_embeddings_txt,
                                                                self.entity_embeddings_img)

        all_entity_list = u.load_entity_list(self.all_triples_file, self.entity_embeddings_txt)
        self.all_entity_list = self.filter_entity(all_entity_list)
        print("load data complete")

    def my_dense(self, x, nr_hidden, scope, reuse=None):
        with tf.variable_scope(scope):
            h = tf.contrib.layers.fully_connected(x, nr_hidden,
                                                  activation_fn=self.activation_function,
                                                  reuse=reuse,
                                                  scope=scope  # , weights_regularizer= self.max_norm_regulizer
                                                  )
            return h

    def build(self):
        # ........... Creating the model
        with tf.name_scope('input'):
            self.r_input = tf.placeholder(dtype=tf.float32, shape=[None, self.relation_structural_embeddings_size],
                                          name="r_input")

            self.h_pos_txt_input = tf.placeholder(dtype=tf.float32,
                                                  shape=[None, self.entity_structural_embeddings_size],
                                                  name="h_pos_txt_input")
            self.h_neg_txt_input = tf.placeholder(dtype=tf.float32,
                                                  shape=[None, self.entity_structural_embeddings_size],
                                                  name="h_neg_txt_input")

            self.h_pos_img_input = tf.placeholder(dtype=tf.float32,
                                                  shape=[None, self.entity_multimodal_embeddings_size],
                                                  name="h_pos_img_input")
            self.h_neg_img_input = tf.placeholder(dtype=tf.float32,
                                                  shape=[None, self.entity_multimodal_embeddings_size],
                                                  name="h_neg_img_input")

            self.t_pos_txt_input = tf.placeholder(dtype=tf.float32,
                                                  shape=[None, self.entity_structural_embeddings_size],
                                                  name="t_pos_txt_input")
            self.t_pos_img_input = tf.placeholder(dtype=tf.float32,
                                                  shape=[None, self.entity_multimodal_embeddings_size],
                                                  name="t_pos_img_input")

            self.t_neg_txt_input = tf.placeholder(dtype=tf.float32,
                                                  shape=[None, self.entity_structural_embeddings_size],
                                                  name="t_neg_txt_input")
            self.t_neg_img_input = tf.placeholder(dtype=tf.float32,
                                                  shape=[None, self.entity_multimodal_embeddings_size],
                                                  name="t_neg_img_input")

            self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        with tf.name_scope('head_relation'):
            # structure
            r_mapped = self.my_dense(self.r_input, self.mapping_size, scope="txt_proj",
                                     reuse=None)
            r_mapped = tf.nn.dropout(r_mapped, self.keep_prob)

            # self.h_pos_txt_mapped = self.my_dense(self.h_pos_txt_input, self.mapping_size,
            #                                  scope="txt_proj", reuse=True)
            self.h_pos_txt_mapped = tf.nn.dropout(self.my_dense(self.h_pos_txt_input, self.mapping_size,
                                             scope="txt_proj", reuse=True), self.keep_prob)

            h_neg_txt_mapped = self.my_dense(self.h_neg_txt_input, self.mapping_size,
                                             scope="txt_proj", reuse=True)
            h_neg_txt_mapped = tf.nn.dropout(h_neg_txt_mapped, self.keep_prob)

            t_pos_txt_mapped = self.my_dense(self.t_pos_txt_input, self.mapping_size,
                                             scope="txt_proj", reuse=True)
            t_pos_txt_mapped = tf.nn.dropout(t_pos_txt_mapped, self.keep_prob)

            t_neg_txt_mapped = self.my_dense(self.t_neg_txt_input, self.mapping_size,
                                             scope="txt_proj", reuse=True)
            t_neg_txt_mapped = tf.nn.dropout(t_neg_txt_mapped, self.keep_prob)

            # Visual
            # h_pos_img_mapped = self.my_dense(self.h_pos_img_input, self.mapping_size, scope="img_proj", reuse=None)
            self.h_pos_img_mapped = tf.nn.dropout(self.my_dense(self.h_pos_img_input, self.mapping_size, scope="img_proj", reuse=None), self.keep_prob)

            h_neg_img_mapped = self.my_dense(self.h_neg_img_input, self.mapping_size, scope="img_proj", reuse=True)
            h_neg_img_mapped = tf.nn.dropout(h_neg_img_mapped, self.keep_prob)

            # Tail image ....
            t_pos_img_mapped = self.my_dense(self.t_pos_img_input, self.mapping_size, scope="img_proj", reuse=True)
            t_pos_img_mapped = tf.nn.dropout(t_pos_img_mapped, self.keep_prob)

            t_neg_img_mapped = self.my_dense(self.t_neg_img_input, self.mapping_size, scope="img_proj", reuse=True)
            t_neg_img_mapped = tf.nn.dropout(t_neg_img_mapped, self.keep_prob)

        with tf.name_scope('cosine'):
            # Head model
            energy_ss_pos = tf.reduce_sum(abs(self.h_pos_txt_mapped + r_mapped - t_pos_txt_mapped), 1, keep_dims=True,
                                          name="pos_s_s")
            energy_ss_neg = tf.reduce_sum(abs(self.h_pos_txt_mapped + r_mapped - t_neg_txt_mapped), 1, keep_dims=True,
                                          name="neg_s_s")

            energy_is_pos = tf.reduce_sum(abs(self.h_pos_img_mapped + r_mapped - t_pos_txt_mapped), 1, keep_dims=True,
                                          name="pos_i_i")
            energy_is_neg = tf.reduce_sum(abs(self.h_pos_img_mapped + r_mapped - t_neg_txt_mapped), 1, keep_dims=True,
                                          name="neg_i_i")

            energy_si_pos = tf.reduce_sum(abs(self.h_pos_txt_mapped + r_mapped - t_pos_img_mapped), 1, keep_dims=True,
                                          name="pos_s_i")
            energy_si_neg = tf.reduce_sum(abs(self.h_pos_txt_mapped + r_mapped - t_neg_img_mapped), 1, keep_dims=True,
                                          name="neg_s_i")

            energy_ii_pos = tf.reduce_sum(abs(self.h_pos_img_mapped + r_mapped - t_pos_img_mapped), 1, keep_dims=True,
                                          name="pos_i_i")
            energy_ii_neg = tf.reduce_sum(abs(self.h_pos_img_mapped + r_mapped - t_neg_img_mapped), 1, keep_dims=True,
                                          name="neg_i_i")

            energy_concat_pos = tf.reduce_sum(
                abs((self.h_pos_txt_mapped + self.h_pos_img_mapped) + r_mapped - (t_pos_txt_mapped + t_pos_img_mapped)), 1,
                keep_dims=True, name="energy_concat_pos")
            energy_concat_neg = tf.reduce_sum(
                abs((self.h_pos_txt_mapped + self.h_pos_img_mapped) + r_mapped - (t_neg_txt_mapped + t_neg_img_mapped)), 1,
                keep_dims=True, name="energy_concat_neg")

            h_r_t_pos = tf.reduce_sum([energy_ss_pos, energy_is_pos, energy_si_pos, energy_ii_pos, energy_concat_pos],
                                      0, name="h_r_t_pos")
            h_r_t_neg = tf.reduce_sum([energy_ss_neg, energy_is_neg, energy_si_neg, energy_ii_neg, energy_concat_neg],
                                      0, name="h_r_t_neg")

            # Tail model

            score_t_t_pos = tf.reduce_sum(abs(t_pos_txt_mapped - r_mapped - self.h_pos_txt_mapped), 1, keep_dims=True,
                                          name="pos_s_s")
            score_t_t_neg = tf.reduce_sum(abs(t_pos_txt_mapped - r_mapped - h_neg_txt_mapped), 1, keep_dims=True,
                                          name="neg_s_s")

            score_i_t_pos = tf.reduce_sum(abs(t_pos_img_mapped - r_mapped - self.h_pos_txt_mapped), 1, keep_dims=True,
                                          name="pos_i_i")
            score_i_t_neg = tf.reduce_sum(abs(t_pos_img_mapped - r_mapped - h_neg_txt_mapped), 1, keep_dims=True,
                                          name="neg_i_i")

            score_t_i_pos = tf.reduce_sum(abs(t_pos_txt_mapped - r_mapped - self.h_pos_img_mapped), 1, keep_dims=True,
                                          name="pos_s_i")
            score_t_i_neg = tf.reduce_sum(abs(t_pos_txt_mapped - r_mapped - h_neg_img_mapped), 1, keep_dims=True,
                                          name="neg_s_i")

            score_i_i_pos = tf.reduce_sum(abs(t_pos_img_mapped - r_mapped - self.h_pos_img_mapped), 1, keep_dims=True,
                                          name="pos_i_i")
            score_i_i_neg = tf.reduce_sum(abs(t_pos_img_mapped - r_mapped - h_neg_img_mapped), 1, keep_dims=True,
                                          name="neg_i_i")

            energy_concat_pos_tail = tf.reduce_sum(
                abs((t_pos_txt_mapped + t_pos_img_mapped) - r_mapped - (self.h_pos_txt_mapped + self.h_pos_img_mapped)), 1,
                keep_dims=True, name="energy_concat_pos_tail")
            energy_concat_neg_tail = tf.reduce_sum(
                abs((t_pos_txt_mapped + t_pos_img_mapped) - r_mapped - (h_neg_txt_mapped + h_neg_img_mapped)), 1,
                keep_dims=True, name="energy_concat_neg_tail")

            t_r_h_pos = tf.reduce_sum(
                [score_t_t_pos, score_i_t_pos, score_t_i_pos, score_i_i_pos, energy_concat_pos_tail], 0,
                name="t_r_h_pos")
            t_r_h_neg = tf.reduce_sum(
                [score_t_t_neg, score_i_t_neg, score_t_i_neg, score_i_i_neg, energy_concat_neg_tail], 0,
                name="t_r_h_neg")

            kbc_loss1 = tf.maximum(0., self.margin - h_r_t_neg + h_r_t_pos)
            kbc_loss2 = tf.maximum(0., self.margin - t_r_h_neg + t_r_h_pos)

            self.kbc_loss = kbc_loss1 + kbc_loss2

            tf.summary.histogram("loss", self.kbc_loss)

        # epsilon= 0.1
        self.optimizer = tf.train.AdamOptimizer().minimize(self.kbc_loss)

        self.summary_op = tf.summary.merge_all()

    def train_worker(self, worker_id):
        print(f"worker {worker_id} start...")
        while (self.epoch < self.training_epochs):
            self.epoch += 1
            ep = self.epoch
            train_data = self.training_data
            np.random.shuffle(train_data)
            training_loss = 0.
            total_batch = int(len(train_data) // self.batch_size + 1)
            for i in range(total_batch):
                start = i * self.batch_size
                end = (i + 1) * self.batch_size
                h_data_txt, h_data_img, r_data, t_data_txt, \
                t_data_img, t_neg_data_txt, t_neg_data_img, h_neg_data_txt, h_neg_data_img = u.get_batch_with_neg_heads_and_neg_tails_multimodal(
                    self.training_data, self.triples_set, self.entity_list, start,
                    end, self.entity_embeddings_txt, self.entity_embeddings_img)
                _, loss, summary = self.sess.run(
                    [self.optimizer, self.kbc_loss, self.summary_op],
                    feed_dict={self.r_input: r_data,
                               self.h_pos_txt_input: h_data_txt,
                               self.h_pos_img_input: h_data_img,

                               self.t_pos_txt_input: t_data_txt,
                               self.t_pos_img_input: t_data_img,

                               self.t_neg_txt_input: t_neg_data_txt,
                               self.t_neg_img_input: t_neg_data_img,

                               self.h_neg_txt_input: h_neg_data_txt,
                               self.h_neg_img_input: h_neg_data_img,

                               self.keep_prob: 1 - self.dropout_ratio  # ,
                               # learning_rate : param.initial_learning_rate
                               })
                # sess.run(clip_all_weights)
                batch_loss = np.sum(loss) / self.batch_size
                training_loss += batch_loss
                # self.writer.add_summary(summary, ep * total_batch + i)

            training_loss = training_loss / total_batch
            if (ep % self.display_step == 0):
                val_loss = self.sess.run([self.kbc_loss],
                                         feed_dict={self.r_input: self.r_data_valid,
                                                    self.h_pos_txt_input: self.h_data_valid_txt,
                                                    self.h_pos_img_input: self.h_data_valid_img,

                                                    self.t_pos_txt_input: self.t_data_valid_txt,
                                                    self.t_pos_img_input: self.t_data_valid_img,

                                                    self.t_neg_txt_input: self.t_neg_data_valid_txt,
                                                    self.t_neg_img_input: self.t_neg_data_valid_img,

                                                    self.h_neg_txt_input: self.h_neg_data_valid_txt,
                                                    self.h_neg_img_input: self.h_neg_data_valid_img,

                                                    self.keep_prob: 1
                                                    })

                val_score = np.sum(val_loss) / len(self.valid_data)
                print(f"worker {worker_id} epoch {ep} loss {str(round(training_loss, 4))} val_loss  {str(round(val_score, 4))}")
            if val_score < self.initial_valid_loss:
                self.saver.save(self.sess, self.model_weights_best_valid_file)
                self.initial_valid_loss = val_score
            # self.saver.save(self.sess, self.model_current_weights_file)
        else:
            print(f"worker {worker_id} epoch {ep} loss {str(round(training_loss, 4))}")


    def train(self):
        # ..... start the training
        self.saver = tf.train.Saver()
        # log_file = open(self.log_file, "w")
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        self.sess.run(tf.global_variables_initializer())
        # Load pre-trained weights if available
        # if os.path.isfile(self.best_valid_model_meta_file):
        #     print("restore the weights", self.checkpoint_best_valid_dir)
        #     saver = tf.train.import_meta_graph(self.best_valid_model_meta_file)
        #     saver.restore(sess, tf.train.latest_checkpoint(self.checkpoint_best_valid_dir))
        # else:
        #     print("no weights to load :(")

        self.writer = tf.summary.FileWriter(self.logs_path, graph=tf.get_default_graph())
        self.initial_valid_loss = 100
        self.epoch = 0
        thread_pool = []
        for i in range(self.thread_num):
            t = threading.Thread(target=self.train_worker, args=(i,))
            thread_pool.append(t)
            t.start()
        for i in range(self.thread_num):
            thread_pool[i].join()
        print("train complete!")


    def test(self):
        print("#Entities", len(self.all_entity_list))
        all_triples = u.load_triples(self.all_triples_file, self.all_entity_list)
        all_test_triples = u.load_triples(self.test_triples_file, self.all_entity_list)
        all_test_triples = self.filter_triples(all_test_triples)
        all_triples = self.filter_triples(all_triples)

        # all_test_triples = all_test_triples[:1000]
        print("#Test triples", len(all_test_triples))  # Triple: head, tail, relation
        self.graph = tf.get_default_graph()
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        with tf.Session(config=sess_config) as sess:
            # print("Model restored from file: %s" % param.current_model_meta_file)
            saver = tf.train.import_meta_graph(self.best_valid_model_meta_file)
            saver.restore(sess, tf.train.latest_checkpoint(self.checkpoint_best_valid_dir))
            test_triple_num = len(all_test_triples)
            avg_rank_raw_tail, avg_rank_raw_head = 0.0, 0.0
            avg_mr_raw_tail, avg_mr_raw_head = 0.0, 0.0
            avg_rank_filter_tail, avg_rank_filter_head = 0.0, 0.0
            avg_mr_filter_tail, avg_mr_filter_head = 0.0, 0.0
            hits_at_10_raw_tail, hits_at_10_raw_head = 0.0, 0.0
            hits_at_3_raw_tail, hits_at_3_raw_head = 0.0, 0.0
            hits_at_1_raw_tail, hits_at_1_raw_head = 0.0, 0.0

            hits_at_10_filter_tail, hits_at_10_filter_head = 0.0, 0.0
            hits_at_3_filter_tail, hits_at_3_filter_head = 0.0, 0.0
            hits_at_1_filter_tail, hits_at_1_filter_head = 0.0, 0.0
            index = 0
            for triple in all_test_triples:
                if (triple[0] in self.entity_embeddings_img and triple[1] in self.entity_embeddings_img
                        and triple[0] in self.entity_embeddings_txt and triple[1] in self.entity_embeddings_txt and
                        triple[2] in self.relation_embeddings):
                    rank_raw_t, rank_filter_t, top_10_t = self.predict_best_tail(triple, all_triples,
                                                                                 self.all_entity_list,
                                                                                 self.entity_embeddings_txt,
                                                                                 self.entity_embeddings_img,
                                                                                 self.relation_embeddings)
                    avg_rank_raw_tail += rank_raw_t
                    avg_mr_raw_tail += 1/rank_raw_t
                    avg_rank_filter_tail += rank_filter_t
                    avg_mr_filter_tail += 1/rank_filter_t
                    if rank_raw_t <= 10:
                        hits_at_10_raw_tail += 1
                    if rank_raw_t <= 3:
                        hits_at_3_raw_tail += 1
                    if rank_raw_t <= 1:
                        hits_at_1_raw_tail += 1

                    if rank_filter_t <= 10:
                        hits_at_10_filter_tail += 1
                    if rank_filter_t <= 3:
                        hits_at_3_filter_tail += 1
                    if rank_filter_t <= 1:
                        hits_at_1_filter_tail += 1

                    rank_raw_h, rank_filter_h, top_10_h  = self.predict_best_head(triple, all_triples,
                                                                                 self.all_entity_list,
                                                                                 self.entity_embeddings_txt,
                                                                                 self.entity_embeddings_img,
                                                                                 self.relation_embeddings)
                    avg_rank_raw_head += rank_raw_h
                    avg_mr_raw_head += 1/rank_raw_h
                    avg_rank_filter_head += rank_filter_h
                    avg_mr_filter_head += 1/rank_filter_h
                    if rank_raw_h <= 10:
                        hits_at_10_raw_head += 1
                    if rank_raw_h <= 3:
                        hits_at_3_raw_head += 1
                    if rank_raw_h <= 1:
                        hits_at_1_raw_head += 1

                    if rank_filter_h <= 10:
                        hits_at_10_filter_head += 1
                    if rank_filter_h <= 3:
                        hits_at_3_filter_head += 1
                    if rank_filter_h <= 1:
                        hits_at_1_filter_head += 1
                index += 1
                if (index % self.display_step == 0):
                    print(f"{index} of {test_triple_num}")
                    print("MA Raw AVG \t MA Filter AVG \t Hits Raw AVG \t Hits Filter AVG")
                    avg_ma_raw = (avg_rank_raw_tail + avg_rank_raw_head) / index / 2
                    avg_ma_filter = (avg_rank_filter_tail + avg_rank_filter_head) / index / 2
                    avg_hits_raw = (hits_at_10_raw_tail + hits_at_10_raw_head) / index / 2
                    avg_hits_filter = (hits_at_10_filter_tail + hits_at_10_filter_head) / index / 2
                    print(str(avg_ma_raw) + "\t" + str(avg_ma_filter) + "\t" + str(avg_hits_raw) + "\t" + str(
                        avg_hits_filter))
            test_triple_num = index
            avg_rank_raw_tail /= test_triple_num
            avg_mr_raw_tail /= test_triple_num
            avg_rank_filter_tail /= test_triple_num
            avg_mr_filter_tail /= test_triple_num
            hits_at_10_raw_tail /= test_triple_num
            hits_at_3_raw_tail /= test_triple_num
            hits_at_1_raw_tail /= test_triple_num
            hits_at_10_filter_tail /= test_triple_num
            hits_at_3_filter_tail /= test_triple_num
            hits_at_1_filter_tail /= test_triple_num

            print("MAR Raw", avg_rank_raw_tail, "MAR Filter", avg_rank_filter_tail)
            print("MR Raw", avg_mr_raw_tail, "MR Filter", avg_mr_filter_tail)
            print("Hits@10 Raw", hits_at_10_raw_tail, "Hits@10 Filter", hits_at_10_filter_tail)
            print("Hits@3 Raw", hits_at_3_raw_tail, "Hits@3 Filter", hits_at_3_filter_tail)
            print("Hits@1 Raw", hits_at_1_raw_tail, "Hits@1 Filter", hits_at_1_filter_tail)


            avg_rank_raw_head /= test_triple_num
            avg_mr_raw_head /= test_triple_num
            avg_mr_filter_head /= test_triple_num
            hits_at_10_raw_head /= test_triple_num
            hits_at_3_raw_head /= test_triple_num
            hits_at_1_raw_head /= test_triple_num
            hits_at_10_filter_head /= test_triple_num
            hits_at_3_filter_head /= test_triple_num
            hits_at_1_filter_head /= test_triple_num

            print("MAR Raw", avg_rank_raw_head, "MAR Filter", avg_rank_filter_head)
            print("MR Raw", avg_mr_raw_head, "MR Filter", avg_mr_filter_head)
            print("Hits@10 Raw", hits_at_10_raw_head, "Hits@10 Filter", hits_at_10_filter_head)
            print("Hits@3 Raw", hits_at_3_raw_head, "Hits@3 Filter", hits_at_3_filter_head)
            print("Hits@1 Raw", hits_at_1_raw_head, "Hits@1 Filter", hits_at_1_filter_head)

        print("+++++++++++++++ Evaluation Summary ++++++++++++++++")
        print("MA Raw Tail \t MA Filter Tail \t Hits Raw Tail \t Hits Filter Tail")
        print(str(avg_rank_raw_tail) + "\t" + str(avg_rank_filter_tail) + "\t" + str(hits_at_10_raw_tail) + "\t" + str(
            hits_at_10_filter_tail))

        print("MA Raw Head \t MA Filter Head \t Hits Raw Head \t Hits Filter Head")
        print(str(avg_rank_raw_head) + "\t" + str(avg_rank_filter_head) + "\t" + str(hits_at_10_raw_head) + "\t" + str(
            hits_at_10_filter_head))

        print("MA Raw AVG \t MA Filter AVG \t Hits Raw AVG \t Hits Filter AVG")
        avg_ma_raw = (avg_rank_raw_tail + avg_rank_raw_head) / 2
        print(f"MA Raw AVG {avg_ma_raw} ")
        avg_mr_raw = (avg_mr_raw_tail + avg_mr_raw_head) / 2
        print(f"MR Raw AVG {avg_mr_raw} ")
        avg_ma_filter = (avg_rank_filter_tail + avg_rank_filter_head) / 2
        print(f"MA Filter AVG {avg_ma_filter} ")
        avg_mr_filter = (avg_mr_filter_tail + avg_mr_filter_head) / 2
        print(f"MR Filter AVG {avg_mr_filter} ")

        avg_hits_raw = (hits_at_10_raw_tail + hits_at_10_raw_head) / 2
        print(f"Hits 10 Raw AVG {avg_hits_raw} ")

        avg_hits3_raw = (hits_at_3_raw_tail + hits_at_3_raw_head) / 2
        print(f"Hits 3 Raw AVG {avg_hits3_raw} ")

        avg_hits1_raw = (hits_at_1_raw_tail + hits_at_1_raw_head) / 2
        print(f"Hits 1 Raw AVG {avg_hits1_raw} ")

        avg_hits_filter = (hits_at_10_filter_tail + hits_at_10_filter_head) / 2
        print(f"Hits 10 Filter AVG {avg_hits_filter} ")

        avg_hits3_filter = (hits_at_3_filter_tail + hits_at_3_filter_head) / 2
        print(f"Hits 3 Filter AVG {avg_hits3_filter} ")

        avg_hits1_filter = (hits_at_1_filter_tail + hits_at_1_filter_head) / 2
        print(f"Hits 1 Filter AVG {avg_hits1_filter} ")

    def train_thread(self, worker_id, data):
        training_loss = 0.
        print(f"thread {worker_id} start")
        total_batch = int(len(self.training_data) // self.batch_size + 1)
        for item in data:
            h_data_txt, h_data_img, r_data, t_data_txt, \
            t_data_img, t_neg_data_txt, t_neg_data_img, \
            h_neg_data_txt, h_neg_data_img = item
            _, loss, summary = self.sess.run(
                [self.optimizer, self.kbc_loss, self.summary_op],
                feed_dict={self.r_input: r_data,
                           self.h_pos_txt_input: h_data_txt,
                           self.h_pos_img_input: h_data_img,

                           self.t_pos_txt_input: t_data_txt,
                           self.t_pos_img_input: t_data_img,

                           self.t_neg_txt_input: t_neg_data_txt,
                           self.t_neg_img_input: t_neg_data_img,

                           self.h_neg_txt_input: h_neg_data_txt,
                           self.h_neg_img_input: h_neg_data_img,

                           self.keep_prob: 1 - self.dropout_ratio  # ,
                           # learning_rate : param.initial_learning_rate
                           })
            # sess.run(clip_all_weights)
            batch_loss = np.sum(loss) / self.batch_size
            training_loss += batch_loss
            # self.writer.add_summary(summary, ep * total_batch + i)

        training_loss = training_loss / total_batch
        if (self.epoch_step % self.display_step == 0):
            total_valid_batch = int(len(self.h_data_valid_txt) // self.batch_size + 1)
            val_loss = 0
            for i in range(total_valid_batch):
                start = i * self.batch_size
                end = (i + 1) * self.batch_size
                val_loss += self.sess.run([self.kbc_loss],
                                          feed_dict={self.r_input: self.r_data_valid[start:end],
                                                     self.h_pos_txt_input: self.h_data_valid_txt[start:end],
                                                     self.h_pos_img_input: self.h_data_valid_img[start:end],

                                                     self.t_pos_txt_input: self.t_data_valid_txt[start:end],
                                                     self.t_pos_img_input: self.t_data_valid_img[start:end],

                                                     self.t_neg_txt_input: self.t_neg_data_valid_txt[start:end],
                                                     self.t_neg_img_input: self.t_neg_data_valid_img[start:end],

                                                     self.h_neg_txt_input: self.h_neg_data_valid_txt[start:end],
                                                     self.h_neg_img_input: self.h_neg_data_valid_img[start:end],

                                                     self.keep_prob: 1
                                                     })

            val_score = np.sum(val_loss) / total_valid_batch
            print(f" epoch {self.epoch_step} loss {str(round(training_loss, 4))} val_loss  {str(round(val_score, 4))}")
            if val_score < self.initial_valid_loss:
                self.saver.save(self.sess, self.model_weights_best_valid_file)
                self.initial_valid_loss = val_score
            # self.saver.save(self.sess, self.model_current_weights_file)
        else:
            print(f"epoch {self.epoch_step} loss {str(round(training_loss, 4))}")


    def consumer_thread(self, worker_id):
        training_loss = 0.
        print(f"thread {worker_id} start")
        total_batch = int(len(self.training_data) // self.batch_size + 1)
        batch_num = 0
        while True:
            try:
                data = self.que.get()
            except:
                print("get error!")
                continue
            if isinstance(data, str) and data == "quit":
                return
            h_data_txt, h_data_img, r_data, t_data_txt, \
            t_data_img, t_neg_data_txt, t_neg_data_img, \
            h_neg_data_txt, h_neg_data_img = data
            _, loss, summary = self.sess.run(
                [self.optimizer, self.kbc_loss, self.summary_op],
                feed_dict={self.r_input: r_data,
                           self.h_pos_txt_input: h_data_txt,
                           self.h_pos_img_input: h_data_img,

                           self.t_pos_txt_input: t_data_txt,
                           self.t_pos_img_input: t_data_img,

                           self.t_neg_txt_input: t_neg_data_txt,
                           self.t_neg_img_input: t_neg_data_img,

                           self.h_neg_txt_input: h_neg_data_txt,
                           self.h_neg_img_input: h_neg_data_img,

                           self.keep_prob: 1 - self.dropout_ratio  # ,
                           # learning_rate : param.initial_learning_rate
                           })
            # sess.run(clip_all_weights)
            batch_loss = np.sum(loss) / self.batch_size
            training_loss += batch_loss
            batch_num += 1
            # self.writer.add_summary(summary, ep * total_batch + i)
            if (batch_num % total_batch == 0):
                # print(f"left batch: {self.que.qsize()}")
                self.epoch_step += 1
                training_loss = training_loss / total_batch
                if (self.epoch_step % self.display_step == 0):
                    total_valid_batch = int(len(self.h_data_valid_txt) // self.batch_size_valid + 1)
                    val_loss = 0
                    for i in range(total_valid_batch):
                        start = i * self.batch_size_valid
                        end = (i + 1) * self.batch_size_valid

                        run_res = self.sess.run([self.kbc_loss],
                                                feed_dict={self.r_input:self.r_data_valid[start:end],
                                                           self.h_pos_txt_input: self.h_data_valid_txt[start:end],
                                                           self.h_pos_img_input: self.h_data_valid_img[start:end],
                                                           self.t_pos_txt_input: self.t_data_valid_txt[start:end],
                                                           self.t_pos_img_input: self.t_data_valid_img[start:end],
                                                           self.t_neg_txt_input: self.t_neg_data_valid_txt[start:end],
                                                           self.t_neg_img_input:self.t_neg_data_valid_img[start:end],
                                                           self.h_neg_txt_input: self.h_neg_data_valid_txt[start:end],
                                                           self.h_neg_img_input: self.h_neg_data_valid_img[start:end],
                                                           self.keep_prob: 1
                                                           })
                        val_loss += np.sum(run_res)

                    val_score = val_loss / int(len(self.h_data_valid_txt))
                    print(f" epoch {self.epoch_step} loss {str(round(training_loss, 4))} val_loss  {str(round(val_score, 4))}")
                if val_score < self.initial_valid_loss:
                    self.saver.save(self.sess, self.model_weights_best_valid_file)
                    self.initial_valid_loss = val_score
                # self.saver.save(self.sess, self.model_current_weights_file)
                else:
                    print(f"thread: {worker_id} epoch {self.epoch_step} loss {str(round(training_loss, 4))}")
                training_loss = 0.


    def consumer(self):
        print("comsumer start")
        thread_pool = ThreadPool(processes=self.thread_num)
        thread_pool.map(self.consumer_thread, [i for i in range(self.thread_num)])
        # while (not self.producer_end_flag):
        #         print(self.que.qsize())
        #         for root, dirs, files in os.walk(self.producer_train_dir):
        #             for file in files:
        #                 try:
        #                     with open(os.path.join(root, file),"rb") as f:
        #                         data = pickle.load(f)
        #                     os.remove(os.path.join(root, file))
        #                     print(f"process {os.path.join(root, file)}...")
        #                     self.epoch_step += 1
        #                     # thread_pool.map_async(self.train_thread,(self.epoch_step+1,data))
        #
        #                 except:
        #                     pass
        #                 break
        #         time.sleep(3)
        thread_pool.close()
        thread_pool.join()
        print("consumer complete")


    def train_multiprocess(self):
        # ..... start the training
        self.saver = tf.train.Saver()
        # log_file = open(self.log_file, "w")
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        self.sess.run(tf.global_variables_initializer())

        self.epoch_step = 0
        self.initial_valid_loss = 100
        pool = Pool(processes=self.process_num)
        self.que = multiprocessing.Manager().Queue()
        self.producer_end_flag = False
        self.total_batch = int(len(self.training_data) // self.batch_size + 1)
        t = threading.Thread(target=self.consumer)
        t.start()

        for i in range(self.total_batch):
            pool.apply_async(producer_worker, (i,
                                               self.training_data, self.triples_set, self.entity_list,
                                               self.entity_embeddings_txt,
                                               self.entity_embeddings_img, self.batch_size, self.producer_train_dir,
                                               self.que))
        pool.close()
        pool.join()
        for i in range(self.thread_num):
            self.que.put("quit")
        print("prodecer complete")
        self.producer_end_flag = True
        t.join()

    def save_multi_entity_embedding(self):
        struct_embedding_multi = {}
        img_imbedding_multi = {}
        for e_name in self.entity_list:
            if e_name in self.entity_embeddings_txt:
                input = np.asarray(self.entity_embeddings_txt[e_name]).reshape((-1,self.entity_structural_embeddings_size))
                # input = tf.to_float(input)
                es = self.sess.run(self.h_pos_txt_mapped,
                                   feed_dict={
                                        self.h_pos_txt_input:input,
                                        self.keep_prob: 1 - self.dropout_ratio
                                   })
                struct_embedding_multi[e_name] = es
            if e_name in self.entity_embeddings_img:
                input = np.asarray(self.entity_embeddings_img[e_name]).reshape(
                    (-1, self.entity_multimodal_embeddings_size))
                # input = tf.to_float(input)
                ei = self.sess.run(self.h_pos_img_mapped,
                                   feed_dict={
                                       self.h_pos_img_input: input,
                                       self.keep_prob: 1.0
                                   })
                img_imbedding_multi[e_name] = ei
        import pickle
        with open(self.embedding_save_path + "mtrl_img_100.pkl","wb") as f_txt:
            pickle.dump(struct_embedding_multi,f_txt)
        with open(self.embedding_save_path + "mtrl_struct_100.pkl","wb") as f_img:
            pickle.dump(img_imbedding_multi,f_img)
        print("ok")

    def load_multi_entity_embedding(self):
        import pickle
        with open(self.embedding_save_path + "mtrl_struct_100.pkl","rb") as f_img:
            img_imbedding_multi = pickle.load(f_img)
        with open(self.embedding_save_path + "mtrl_img_100.pkl","rb") as f_txt:
            struct_embedding_multi = pickle.load(f_txt)

        print("load_multi_entity_embedding conplete!")
        return struct_embedding_multi,img_imbedding_multi

def producer_worker(work_id, training_data, triples_set, entity_list, entity_embedding_txt, entity_embedding_img,
                    batch_size, producer_train_dir, que):
    # print(f"producer {work_id} start")
    np.random.shuffle(training_data)
    total_batch = int(len(training_data) // batch_size + 1)
    for i in range(total_batch):
        start = i * batch_size
        end = (i + 1) * batch_size
        que.put(u.get_batch_with_neg_heads_and_neg_tails_multimodal(training_data, triples_set, entity_list, start, end,
                                                                    entity_embedding_txt, entity_embedding_img))
        while (que.qsize() >= 8000):
            time.sleep(3)
    print(f"producer {work_id} complete")


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    ik = mtrl()
    # ik.load_multi_entity_embedding()
