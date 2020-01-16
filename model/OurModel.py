# coding: utf-8
from model.BaseModel import BaseModel
import tensorflow as tf



class OurModel(BaseModel):
    """我们的模型"""

    def build(self):
        mention_img_mapped, mention_lin_mapped, \
        entity_img_pos_mapped, entity_struct_pos_mapped, entity_lin_pos_mapped, \
        entity_img_neg_mapped, entity_struct_neg_mapped, entity_lin_neg_mapped, \
        e_pos_rnn_output, e_pos_rnn_forward, e_pos_rnn_backward, \
        e_neg_rnn_output, e_neg_rnn_forward, e_neg_rnn_backward, \
        mention_rnn_output, mention_rnn_forward, mention_rnn_backward = self.shared_underlaying()

        # 将mention上下文向量,实体desc向量
        pos_attn_f, self.m_pf_atten = self.interactive_attention.get_attn(mention_rnn_output[0],
                                                                          [self.input_entity_struct_pos])
        pos_attn_b, self.m_pb_atten = self.interactive_attention.get_attn(mention_rnn_output[1],
                                                                          [self.input_entity_struct_pos])
        neg_attn_f, self.m_nf_atten = self.interactive_attention.get_attn(mention_rnn_output[0],
                                                                          [self.input_entity_struct_neg])
        neg_attn_b, self.m_nb_atten = self.interactive_attention.get_attn(mention_rnn_output[1],
                                                                          [self.input_entity_struct_neg])

        pos_mention_attention_dense = self.my_dense(tf.concat([pos_attn_f, pos_attn_b], axis=-1), self.config.hidden_size,
                                  scope="mention_attention_dense")
        neg_mention_attention_dense = self.my_dense(tf.concat([neg_attn_f, neg_attn_b], axis=-1), self.config.hidden_size,
                                  scope="mention_attention_dense")

        mention_keys = [mention_rnn_forward, mention_rnn_backward]
        pos_keys = [e_pos_rnn_forward, e_pos_rnn_backward, entity_struct_pos_mapped]
        neg_keys = [e_neg_rnn_forward, e_neg_rnn_backward, entity_struct_neg_mapped]
        if self.use_lin_embedding:
            mention_keys.append(mention_lin_mapped)
            pos_keys.append(entity_lin_pos_mapped)
            neg_keys.append(entity_lin_neg_mapped)
        if self.use_mention_img_embedding:
            mention_keys.append(mention_img_mapped)
        if self.use_entity_img_embedding:
            pos_keys.append(entity_img_pos_mapped)
            neg_keys.append(entity_img_neg_mapped)

        mention_concat = tf.concat(mention_keys, axis=-1)
        entity_pos_concat = tf.concat(pos_keys, axis=-1)
        entity_neg_concat = tf.concat(neg_keys, axis=-1)
        mention_trans = tf.transpose(
            tf.reshape(mention_concat, [-1, self.config.embedding_dim, len(mention_keys)]), perm=[0, 2, 1])
        entity_pos_trans = tf.transpose(
            tf.reshape(entity_pos_concat, [-1, self.config.embedding_dim, len(pos_keys)]), perm=[0, 2, 1])
        entity_neg_trans = tf.transpose(
            tf.reshape(entity_neg_concat, [-1, self.config.embedding_dim, len(neg_keys)]), perm=[0, 2, 1])

        mention_trans_attn, self.mention_self_attn_weights = self.self_attention.get_self_attn(mention_trans,mention_keys)
        entity_pos_trans_attn, self.entity_self_attn_weights = self.self_attention.get_self_attn(entity_pos_trans,pos_keys)
        entity_neg_trans_attn, _ = self.self_attention.get_self_attn(entity_neg_trans, neg_keys)
        if self.use_self_attention:
            mention_trans = mention_trans
            entity_pos_trans = entity_pos_trans
            entity_neg_trans = entity_neg_trans

        m_pos_attn, self.m_p_atten = self.interactive_attention.get_attn(entity_pos_trans, mention_keys)
        pos_m_attn, self.p_m_atten = self.interactive_attention.get_attn(mention_trans, pos_keys)
        m_neg_attn, _ = self.interactive_attention.get_attn(entity_neg_trans, mention_keys)
        neg_m_attn, _ = self.interactive_attention.get_attn(mention_trans, neg_keys)
        mention_reshape = tf.reshape(mention_trans, [-1, len(mention_keys)*self.config.embedding_dim])
        pos_reshape = tf.reshape(entity_pos_trans, [-1, len(pos_keys)*self.config.embedding_dim])
        neg_reshape = tf.reshape(entity_neg_trans,[-1, len(neg_keys) * self.config.embedding_dim])

        pos_score_concat = [mention_reshape, pos_reshape]
        neg_score_concat = [mention_reshape, neg_reshape]

        if (self.use_interactive_attention):
            pos_score_concat += [m_pos_attn, pos_m_attn]
            neg_score_concat += [m_neg_attn, neg_m_attn]

        if self.use_mention_attention:
            pos_score_concat.append(pos_mention_attention_dense)
            neg_score_concat.append(neg_mention_attention_dense)

        if self.use_fertures:
            feature_pos_tile = tf.tile(self.input_entity_feature_pos, [1, 5])
            pos_score_concat.append(feature_pos_tile)
            feature_neg_tile = tf.tile(self.input_entity_feature_neg, [1, 5])
            neg_score_concat.append(feature_neg_tile)
            pos_score_concat = tf.concat(pos_score_concat, axis=-1)
            neg_score_concat = tf.concat(neg_score_concat, axis=-1)

        pos_dense = self.my_dense(tf.concat(pos_score_concat, axis=-1), self.config.hidden_size*2, scope="score_dense")
        neg_dense = self.my_dense(tf.concat(neg_score_concat, axis=-1), self.config.hidden_size*2, scope="score_dense")
        pos_dense = tf.nn.dropout(pos_dense, self.keep_prob)
        neg_dense = tf.nn.dropout(neg_dense, self.keep_prob)

        self.energy_pos = tf.nn.tanh(tf.nn.dropout(self.score_dense(pos_dense), self.keep_prob), name="energy_pos")
        self.energy_neg = tf.nn.tanh(tf.nn.dropout(self.score_dense(neg_dense), self.keep_prob), name="energy_neg")



        with tf.name_scope("optimize"):
            # 优化器
            print(f"the margin is {self.config.margin}")
            self.kbc_loss = tf.reduce_sum(tf.maximum(0., self.config.margin - self.energy_pos + self.energy_neg))
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.kbc_loss)


if __name__ == "__main__":
    from util.data_util import DataUtil
    from util.config import Config_Util
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    config_util = Config_Util()
    data_helper = DataUtil(config_util)
    data_helper.load_word_embedding()

    model = OurModel(config_util)
    model.use_self_attention = True
    model.use_interactive_attention =True
    model.init_word_embedding(data_helper.word_embedding)
    model.build()
