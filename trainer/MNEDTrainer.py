# coding: utf-8

import tensorflow as tf
import os
import time
import numpy as np
import pickle

def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

class MNEDTrainer(object):

    def __init__(self, data_helper, model, section_save_path):

        self.data_helper = data_helper
        self.config_util = data_helper.config
        self.compute_attention = False
        self.section_save_path = section_save_path
        self.model = model
        self.remind = ""
        self.model_name = model.model_name

        if self.model.use_interactive_attention :
            self.model_name = "External_" +self.model_name
            self.remind += "\t Use External Attention"
        if self.model.use_self_attention:
            self.model_name = 'Internal_' +self.model_name
            self.remind += "\t Use Internal Attention"
        if self.model.use_fertures:
            self.model_name = 'Features_' +self.model_name
            self.remind += "\t Use Fratures"
        if self.model.use_mention_attention:
            self.model_name = 'Mentionattention_' +self.model_name
            self.remind += "\t Use Mention Attention"
        if self.model.use_mention_img_embedding:
            self.model_name = 'Mentionimg_' + self.model_name
            self.remind += "\t Use Mention Img"
        if self.model.use_entity_img_embedding:
            self.model_name = 'Entityimg_' + self.model_name
            self.remind += "\t Use Entity Img"
        if self.model.use_tranE_struct_embedding:
            self.model_name = 'TransE_' + self.model_name
            self.remind += "\t Use TransE"

        print(f"model name is {self.model_name}")

    def load_train_val_data_itr(self,stage = "train"):
        """
        加载训练及验证数据
        :return:
        """
        print(f"Loading {stage} data...")
        start_time = time.time()
        for data in self.data_helper.process_train_file_itr_MNED(
            self.config_util.input_dir + f"{stage}_candidate_json.txt",self.config_util.seq_length):
            time_dif = self.data_helper.get_time_dif(start_time)
            print("Time usage:", time_dif)
            yield data



    def config_tensorboard(self):
        # 配置Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
        print("Configuring TensorBoard and Saver...")
        tensorboard_dir = self.config_util.tensorboard_dir
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)
        tf.summary.scalar("kbc_loss", self.model.kbc_loss)
        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter(tensorboard_dir)
        return writer, merged_summary

    def evaluate(self, sess,  val_mention_imgs, val_mention_contents, val_mention_lins, \
               val_pos_imgs, val_pos_structs, val_pos_descs, val_pos_transEs, val_pos_lins, val_pos_features, \
               val_neg_imgs, val_neg_structs, val_neg_descs, val_neg_transEs, val_neg_lins, val_neg_features):
        data_len = len(val_mention_imgs)

        batch_eval = self.data_helper.batch_iter([val_mention_imgs, val_mention_contents, val_mention_lins, \
               val_pos_imgs, val_pos_structs, val_pos_descs, val_pos_transEs, val_pos_lins, val_pos_features, \
               val_neg_imgs, val_neg_structs, val_neg_descs, val_neg_transEs, val_neg_lins, val_neg_features],
                                                 self.config_util.val_batch_size)
        total_loss = 0.0
        for mention_imgs_batch, mention_contents_batch, mention_lins_batch, \
            pos_imgs_batch, pos_structs_batch, pos_descs_batch, pos_transEs_batch, pos_lins_batch, pos_features_batch, \
            neg_imgs_batch, neg_structs_batch, neg_descs_batch, neg_transEs_batch, neg_lins_batch, neg_features_batch in batch_eval:

            # if batch_index >= 100:
            #     break
            if self.model.use_tranE_struct_embedding:
                neg_structs_batch = neg_transEs_batch
                pos_structs_batch = pos_transEs_batch
            feed_dict = {
                self.model.mention_context: mention_contents_batch,
                self.model.mention_lin: mention_lins_batch,

                self.model.input_entity_struct_pos: pos_structs_batch,
                self.model.input_entity_lin_pos: pos_lins_batch,
                self.model.input_entity_desc_pos: pos_descs_batch,
                self.model.input_entity_feature_pos: pos_features_batch,

                self.model.input_entity_struct_neg: neg_structs_batch,
                self.model.input_entity_lin_neg: neg_lins_batch,
                self.model.input_entity_desc_neg: neg_descs_batch,
                self.model.input_entity_feature_neg: neg_features_batch,

                self.model.keep_prob: self.config_util.dropout_keep_prob
            }
            if self.data_helper.mention_img_embedding:
                feed_dict[self.model.mention_img] = mention_imgs_batch
                feed_dict[self.model.input_entity_img_pos] = pos_imgs_batch
                feed_dict[self.model.input_entity_img_neg] = neg_imgs_batch

            loss = sess.run(self.model.kbc_loss, feed_dict=feed_dict)
            total_loss += loss
        return total_loss / data_len

    def train(self):
        sec_save_path = self.section_save_path + "train_sections/"
        val_pk_file = self.section_save_path + "val.pkl"

        if not os.path.exists(sec_save_path):
            os.makedirs(sec_save_path)
        if(not os.path.exists(sec_save_path + "0_0_seg.pkl")):
            # 预处理数据
           for index, data in enumerate(self.load_train_val_data_itr()):
                train_mention_imgs, train_mention_contents, train_mention_lins,\
                train_pos_imgs, train_pos_structs, train_pos_descs,  train_pos_transEs, train_pos_lins , train_pos_features,\
                train_neg_imgs, train_neg_structs, train_neg_descs,  train_neg_transEs, train_neg_lins, train_neg_features = \
                data["mention_imgs"],data["mention_contents"],data["mention_lins"], \
                data["pos_imgs"],data["pos_txts"],data["pos_descs"],data["pos_transEs"],data["pos_lins"],data["pos_features"],\
                data["neg_imgs"],data["neg_txts"],data["neg_descs"],data["neg_transEs"],data["neg_lins"],data["neg_features"]


                data_arr = [train_mention_imgs, train_mention_contents, train_mention_lins,\
                train_pos_imgs, train_pos_structs, train_pos_descs,  train_pos_transEs, train_pos_lins , train_pos_features,\
                train_neg_imgs, train_neg_structs, train_neg_descs,  train_neg_transEs, train_neg_lins, train_neg_features]
                self.data_helper.section_save(data_arr, 100000, sec_save_path,index)
                print(f"segment index: {index} ok")

           print("load train data ok")

           for index, data in enumerate(self.load_train_val_data_itr("val")):
               val_mention_imgs, val_mention_contents, val_mention_lins, \
               val_pos_imgs, val_pos_structs, val_pos_descs, val_pos_transEs, val_pos_lins, val_pos_features, \
               val_neg_imgs, val_neg_structs, val_neg_descs, val_neg_transEs, val_neg_lins, val_neg_features = \
                   data["mention_imgs"], data["mention_contents"], data["mention_lins"], \
                   data["pos_imgs"], data["pos_txts"], data["pos_descs"], data["pos_transEs"], data["pos_lins"], \
                   data["pos_features"], \
                   data["neg_imgs"], data["neg_txts"], data["neg_descs"], data["neg_transEs"], data["neg_lins"], \
                   data["neg_features"]

               with open(val_pk_file, "wb") as pf:
                    pickle.dump((val_mention_imgs, val_mention_contents, val_mention_lins, \
                   val_pos_imgs, val_pos_structs, val_pos_descs, val_pos_transEs, val_pos_lins, val_pos_features, \
                   val_neg_imgs, val_neg_structs, val_neg_descs, val_neg_transEs, val_neg_lins, val_neg_features), pf)
               print(f"val segment index: {index} ok")
               print("load val data ok")

        else:
            with open(val_pk_file,"rb") as pf:
                val_mention_imgs, val_mention_contents, val_mention_lins, \
                val_pos_imgs, val_pos_structs, val_pos_descs, val_pos_transEs, val_pos_lins, val_pos_features, \
                val_neg_imgs, val_neg_structs, val_neg_descs, val_neg_transEs, val_neg_lins, val_neg_features = pickle.load(pf)

        # 配置tensorboard
        writer, merged_summary = self.config_tensorboard()

        # 配置Saver
        saver = tf.train.Saver()
        if not os.path.exists(self.config_util.save_dir+"model/"+self.model_name):
            os.makedirs(self.config_util.save_dir+"model/"+self.model_name)

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        # config.gpu_options.per_process_gpu_memory_fraction = 0.5
        session = tf.Session(config=config)
        # 创建session
        session.run(tf.global_variables_initializer())
        writer.add_graph(session.graph)

        print('Training and evaluating...')
        start_time = time.time()
        # 总批次
        total_batch = 0
        # 最佳验证集准确率
        best_loss_val = 1000000
        # early stopping的标志位
        flag = False
        for epoch in range(self.config_util.num_epochs):
            print('Epoch:', epoch + 1)
            train_loss_sum =0
            per_index = 0
            batch_train = self.data_helper.section_batch_iter(sec_save_path,is_random=True,batch_size=self.config_util.batch_size)
            batch_index = 0
            for mention_imgs_batch, mention_contents_batch, mention_lins_batch,\
                pos_imgs_batch, pos_structs_batch, pos_descs_batch,  pos_transEs_batch, pos_lins_batch , pos_features_batch,\
                neg_imgs_batch, neg_structs_batch, neg_descs_batch,  neg_transEs_batch, neg_lins_batch, neg_features_batch in batch_train:

                total_batch += 1
                batch_index += 1
                per_index+=1
                # if batch_index >= 100:
                #     break
                if self.model.use_tranE_struct_embedding:
                    neg_structs_batch = neg_transEs_batch
                    pos_structs_batch = pos_transEs_batch
                feed_dict = {
                    self.model.mention_context: mention_contents_batch,
                    self.model.mention_lin: mention_lins_batch,

                    self.model.input_entity_struct_pos: pos_structs_batch,
                    self.model.input_entity_lin_pos: pos_lins_batch,
                    self.model.input_entity_desc_pos:pos_descs_batch,
                    self.model.input_entity_feature_pos : pos_features_batch,

                    self.model.input_entity_struct_neg: neg_structs_batch,
                    self.model.input_entity_lin_neg: neg_lins_batch,
                    self.model.input_entity_desc_neg: neg_descs_batch,
                    self.model.input_entity_feature_neg: neg_features_batch,

                    self.model.keep_prob: self.config_util.dropout_keep_prob
                }
                if self.data_helper.mention_img_embedding:
                    feed_dict[self.model.mention_img] = mention_imgs_batch
                    feed_dict[self.model.input_entity_img_pos] = pos_imgs_batch
                    feed_dict[self.model.input_entity_img_neg] = neg_imgs_batch

                # 将训练结果写入tensorboard scalar
                if total_batch % self.config_util.save_per_batch == 0:
                    s = session.run(merged_summary, feed_dict=feed_dict)
                    writer.add_summary(s, total_batch)

                # 输出在训练集和验证集上的性能
                if total_batch % self.config_util.print_per_batch == 0:
                    feed_dict[self.model.keep_prob] = 1.0
                    loss_val = self.evaluate(session,  val_mention_imgs, val_mention_contents, val_mention_lins, \
               val_pos_imgs, val_pos_structs, val_pos_descs, val_pos_transEs, val_pos_lins, val_pos_features, \
               val_neg_imgs, val_neg_structs, val_neg_descs, val_neg_transEs, val_neg_lins, val_neg_features)
                    self.group_test(session)
                    train_loss_sum =0
                    per_index = 1
                    # 保存最好结果
                    if loss_val < best_loss_val:
                        best_loss_val = loss_val
                        saver.save(sess=session, save_path=self.config_util.save_dir+"model/"+self.model_name)
                        improved_str = '*'
                    else:
                        improved_str = ''

                    time_dif = self.data_helper.get_time_dif(start_time)
                    msg = f'Iter: {total_batch}, Val Loss: {loss_val}, Time: {time_dif} {improved_str}'
                    print(msg)
                    print("*" * 20)

                # 对loss进行优化
                run_result = session.run([self.model.kbc_loss,self.model.optim], feed_dict=feed_dict)  # 运行优化
                train_loss_sum += run_result[0]
                print(f"\r\t epoch: {epoch + 1}/{self.config_util.num_epochs}    batch: {batch_index}/...  loss: {run_result[0]/self.config_util.batch_size}  avg loss: {train_loss_sum/per_index/self.config_util.batch_size}",end='')

                # #验证集正确率长期不提升，提前结束训练
                # if total_batch - last_improved > self.config_util.require_improvement:
                #     print("No optimization for a long time, auto-stopping...")
                #     flag = True
                #     break
            self.group_test(session)
            # early stopping
            if flag:
                break

    def group_test(self,session = None):
        print(self.remind)
        pk_file = self.section_save_path + "test.pkl"
        if (not os.path.exists(pk_file)):
            # 预处理数据
            data = self.data_helper.process_test_file_itr_MNED(
                self.config_util.input_dir + f"test_candidate_json.txt",self.config_util.seq_length)
            val_mention_imgs, val_mention_contents, val_mention_lins, \
            val_pos_imgs, val_pos_structs, val_pos_descs, val_pos_transEs, val_pos_lins, val_pos_features,group_list = \
                data["mention_imgs"], data["mention_contents"], data["mention_lins"], \
                data["pos_imgs"], data["pos_txts"], data["pos_descs"], data["pos_transEs"], data["pos_lins"], \
                data["pos_features"],data["group_list"]

            with open(pk_file, "wb") as pf:
                pickle.dump((val_mention_imgs, val_mention_contents, val_mention_lins, \
            val_pos_imgs, val_pos_structs, val_pos_descs, val_pos_transEs, val_pos_lins, val_pos_features,group_list), pf)
        else:
            with open(pk_file, "rb") as pf:
                val_mention_imgs, val_mention_contents, val_mention_lins, \
                val_pos_imgs, val_pos_structs, val_pos_descs, val_pos_transEs, val_pos_lins, val_pos_features, group_list = pickle.load(pf)
        if(not session):
            # 创建session
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            # config.gpu_options.per_process_gpu_memory_fraction = 0.5
            session = tf.Session(config=config)
            saver = tf.train.Saver()
            # 读取训练好的模型
            saver.restore(sess=session, save_path=self.config_util.save_dir+"model/"+self.model_name)  # 读取保存的模型

        # 保存预测结果
        y_pred_cls = np.zeros(shape=[len(val_pos_descs)], dtype=np.float32)
        batch_test = self.data_helper.batch_iter([val_mention_imgs, val_mention_contents, val_mention_lins, \
                val_pos_imgs, val_pos_structs, val_pos_descs, val_pos_transEs, val_pos_lins, val_pos_features],
                                                 self.config_util.val_batch_size,
                                                 is_random=False)
        index = 0
        m_s,e_s,m_i,e_i = [],[],[],[]
        for mention_imgs_batch, mention_contents_batch, mention_lins_batch, \
                imgs_batch, structs_batch, descs_batch, transEs_batch, lins_batch, features_batch in batch_test:
            if self.model.use_tranE_struct_embedding:
                structs_batch = transEs_batch
            feed_dict = {
                self.model.mention_context: mention_contents_batch,
                self.model.mention_lin: mention_lins_batch,

                self.model.input_entity_desc_pos: descs_batch,
                self.model.input_entity_struct_pos: structs_batch,
                self.model.input_entity_lin_pos: lins_batch,
                self.model.input_entity_feature_pos: features_batch,
                self.model.keep_prob: 1.0
            }
            if self.data_helper.mention_img_embedding:
                feed_dict[self.model.mention_img] = mention_imgs_batch
                feed_dict[self.model.input_entity_img_pos] = imgs_batch
            start_id = index * self.config_util.val_batch_size
            end_id = min((index + 1) * self.config_util.val_batch_size, len(val_pos_descs))
            index += 1
            fd = [self.model.energy_pos]
            if self.show_attention:
                fd+= [self.model.mention_self_attn_weights,self.model.entity_self_attn_weights,
                               self.model.m_p_atten,self.model.p_m_atten]
            y_p = session.run(fd, feed_dict=feed_dict)
            y_pred_cls[start_id:end_id] = y_p[0].reshape(-1)

            if self.show_attention:
                a_self_m = np.squeeze(np.array(y_p[1]))
                m_s.append(a_self_m)

                a_self_e = np.squeeze(np.array(y_p[2]))
                e_s.append(a_self_e)

                a_inter_m = np.squeeze(np.array(y_p[3]))
                m_i.append(a_inter_m)

                a_inter_e = np.squeeze(np.array(y_p[4]))
                e_i.append(a_inter_e)

        test_index = 0
        skip_num = 0
        mrr = 0
        hit_1, hit_3, hit_5, hit_10 = 0, 0, 0, 0
        group_1 =0
        for group_len in group_list:
            group_test = y_pred_cls[test_index:test_index + group_len]

            test_index += group_len
            if group_len ==1:
                group_1+=1
            tmp_dict = {}
            for index, item in enumerate(group_test):
                tmp_dict[index] = item

            tmp_dict = sorted(tmp_dict.items(),key=lambda x: x[1], reverse=True)
            ordered_index = [item[0] for item in tmp_dict]
            if(not  ordered_index):
                print("err")
            if (0 == ordered_index[0]):
                hit_1 += 1
            if (0 in ordered_index[:3]):
                hit_3 += 1
            if (0 in ordered_index[:5]):
                hit_5 += 1
            if (0 in ordered_index[:10]):
                hit_10 += 1
            mrr += 1.0/(ordered_index.index(0)+1)
        groups = len(group_list) - skip_num
        print(f"hit@1: {hit_1 / groups}  hit@3: {hit_3 / groups} hit@5: {hit_5 / groups}  hit@10: {hit_10 / groups}  mrr: {mrr / groups}")
        if self.show_attention:
            attn_list = [np.mean(np.concatenate(m_s, axis=0), axis=0), \
                         np.mean(np.concatenate(e_s, axis=0), axis=0), \
                         np.mean(np.concatenate(m_i, axis=0), axis=0), \
                         np.mean(np.concatenate(e_i, axis=0), axis=0)]
            print("m_s, e_s, m_i, e_i :")
            print(attn_list)
        print(group_1/len(group_list))
