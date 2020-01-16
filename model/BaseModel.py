# coding: utf-8
import tensorflow as tf
from tensorflow.python.ops import math_ops

class InteractiveScaleCosineAttention(tf.keras.Model):
    """
       交互attention类
       执行交互操作
   """
    def __init__(self, units):
        super(InteractiveScaleCosineAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def cosine(self,y_, y, axis=-1):
        y_ = tf.nn.l2_normalize(y_, axis=axis)
        y = tf.nn.l2_normalize(y, axis=axis)
        return -math_ops.reduce_sum(y_ * y, axis=axis)

    def get_attn(self, features, keys):
        concat_context_vector = []
        concat_attention_weights = []
        for key in keys:
            hidden_with_time_axis = tf.expand_dims(key, 1)
            # score = self.cosine(self.W1(features),self.W2(hidden_with_time_axis),axis = -1)
            score = self.cosine(features,hidden_with_time_axis,axis = -1)
            attention_weights = tf.nn.softmax(score, axis=1)
            attention_weights = tf.expand_dims(attention_weights, 2)
            context_vector = attention_weights * features
            concat_context_vector.append(context_vector)
            concat_attention_weights.append(attention_weights)
        concat_context_vector = tf.convert_to_tensor(concat_context_vector)
        concat_context_vector = tf.reduce_sum(concat_context_vector, axis=0)
        concat_attention_weights = tf.convert_to_tensor(concat_attention_weights)
        concat_attention_weights = tf.reduce_sum(concat_attention_weights,axis=0)
        reshape_size = concat_context_vector.get_shape().as_list()
        return tf.reshape(concat_context_vector,[-1,reshape_size[1]*reshape_size[2]]),  tf.squeeze(concat_attention_weights)




class SelfScaleCosineAttention(tf.keras.Model):
    """
       交互attention类
       执行交互操作
   """
    def __init__(self, units):
        super(SelfScaleCosineAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def cosine(self,y_, y, axis=-1):
        y_ = tf.nn.l2_normalize(y_, axis=axis)
        y = tf.nn.l2_normalize(y, axis=axis)
        return -math_ops.reduce_sum(y_ * y, axis=axis)


    def get_self_attn(self, features, keys):
        """
            :param features: attention的key 和 value值
            :param querys: attention的 query值
            :return: attention权重列表和加权后的上文向量
        """
        concat_attention_weights = []
        for key in keys:
            hidden_with_time_axis = tf.expand_dims(key, 1)
            # score = self.cosine(self.W1(features),self.W2(hidden_with_time_axis),axis = -1)
            score = self.cosine(features,hidden_with_time_axis,axis = -1)
            attention_weights = tf.nn.softmax(score, axis=1)
            attention_weights = tf.expand_dims(attention_weights, 2)
            concat_attention_weights.append(attention_weights)
        concat_attention_weights = tf.convert_to_tensor(concat_attention_weights)
        concat_attention_weights = tf.reduce_sum(concat_attention_weights, axis=0)
        self_attn_res = features * concat_attention_weights
        return self_attn_res,  tf.squeeze(concat_attention_weights)


class BaseModel(object):
    """我们的模型"""

    def __init__(self, config):
        # 是否使用双向交互attention
        self.use_interactive_attention = False
        # 是否使用自交互attention
        self.use_self_attention = False
        # 是否适应输入mention上下文的embedding
        self.use_mention_img_embedding = False
        # 是否使用实体图片的embedding
        self.use_entity_img_embedding = False
        # 是否使用结构化的embedding
        self.use_tranE_struct_embedding = False
        self.config = config
        self.map_flag = True
        self.use_lin_embedding =True
        self.use_fertures = True
        self.use_mention_attention = True
        self.feature_num = 12
        self.interactive_attention = InteractiveScaleCosineAttention(self.config.attention_size)
        self.self_attention = SelfScaleCosineAttention(self.config.attention_size)
        self.model_name = "base_model"
        self.input()

    def input(self):
        """
           模型构造函数
       """
        self.score_dense = tf.keras.layers.Dense(1)
        # 输入的数据
        # mention相关
        self.mention_context = tf.placeholder(tf.int32, [None, self.config.seq_length], name='mention_context')
        self.mention_img = tf.placeholder(tf.float32, [None, 4096], name='mention_img')
        self.mention_lin = tf.placeholder(tf.float32, [None, self.config.embedding_dim], name='mention_lin')

        # 正例
        self.input_entity_desc_pos = tf.placeholder(tf.int32, [None, self.config.seq_length],
                                                    name='input_entity_desc_pos')
        self.input_entity_feature_pos = tf.placeholder(tf.float32, [None, self.feature_num],name='input_entity_feature_pos')
        self.input_entity_img_pos = tf.placeholder(tf.float32, [None, self.config.embedding_dim],
                                                   name='input_entity_img_pos')
        self.input_entity_struct_pos = tf.placeholder(tf.float32, [None, self.config.embedding_dim],
                                                      name='input_entity_struct_pos')
        self.input_entity_lin_pos = tf.placeholder(tf.float32, [None, self.config.embedding_dim],
                                                   name='input_entity_lin_pos')

        # 负例
        self.input_entity_desc_neg = tf.placeholder(tf.int32, [None, self.config.seq_length],
                                                    name='input_entity_desc_neg')
        self.input_entity_feature_neg = tf.placeholder(tf.float32, [None, self.feature_num],name='input_entity_feature_neg')
        self.input_entity_img_neg = tf.placeholder(tf.float32, [None, self.config.embedding_dim],
                                                   name='input_entity_img_neg')
        self.input_entity_struct_neg = tf.placeholder(tf.float32, [None, self.config.embedding_dim],
                                                      name='input_entity_struct_neg')
        self.input_entity_lin_neg = tf.placeholder(tf.float32, [None, self.config.embedding_dim],
                                                   name='input_entity_lin_neg')

        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    def init_word_embedding(self,word_embeddings = None):
        with tf.device('/cpu:0'):
            # 不使用预训练词向量
            if not self.config.is_pre_train_embed:
                print("not use pre trained word embedding")
                self.embedding = tf.get_variable(
                    name="embedding", dtype=tf.float32,
                    shape=[self.config.vocab_size, self.config.embedding_dim]
                )
            # 使用预训练词向量
            else:
                embedding = tf.Variable(
                    word_embeddings, name="embedding",
                    dtype=tf.float32, trainable=False)
            self.embedding = embedding

    def shared_underlaying(self):
        embedding_context = tf.nn.embedding_lookup(self.embedding, self.mention_context)
        embedding_entity_desc_neg = tf.nn.embedding_lookup(self.embedding, self.input_entity_desc_neg)
        embedding_entity_desc_pos = tf.nn.embedding_lookup(self.embedding, self.input_entity_desc_pos)

        with tf.name_scope("map"):
            mention_img_mapped = self.my_dense(self.mention_img, self.config.hidden_size, scope="mention_img_mapped")
            if self.map_flag:
                mention_lin_mapped = self.my_dense(self.mention_lin, self.config.hidden_size,
                                                   scope="mention_lin_mapped")
                entity_img_pos_mapped = self.my_dense(self.input_entity_img_pos, self.config.hidden_size,
                                                      scope="entity_img_mapped")
                entity_struct_pos_mapped = self.my_dense(self.input_entity_struct_pos, self.config.hidden_size,
                                                         scope="entity_struct_mapped")
                entity_lin_pos_mapped = self.my_dense(self.input_entity_lin_pos, self.config.hidden_size,
                                                      scope="entity_lin_mapped")
                entity_img_neg_mapped = self.my_dense(self.input_entity_img_neg, self.config.hidden_size,
                                                      scope="entity_img_mapped")
                entity_struct_neg_mapped = self.my_dense(self.input_entity_struct_neg, self.config.hidden_size,
                                                         scope="entity_struct_mapped")
                entity_lin_neg_mapped = self.my_dense(self.input_entity_lin_neg, self.config.hidden_size,
                                                      scope="entity_lin_mapped")
            else:
                mention_lin_mapped = self.mention_lin
                entity_img_pos_mapped = self.input_entity_img_pos
                entity_struct_pos_mapped = self.input_entity_struct_pos
                entity_lin_pos_mapped = self.input_entity_lin_pos
                entity_img_neg_mapped = self.input_entity_img_neg
                entity_struct_neg_mapped = self.input_entity_struct_neg
                entity_lin_neg_mapped = self.input_entity_lin_neg

            mention_img_mapped = tf.nn.l2_normalize(mention_img_mapped, dim=-1)
            mention_lin_mapped = tf.nn.l2_normalize(mention_lin_mapped + self.mention_lin, dim=-1)
            entity_img_pos_mapped = tf.nn.l2_normalize(entity_img_pos_mapped + self.input_entity_img_pos, dim=-1)
            entity_struct_pos_mapped = tf.nn.l2_normalize(entity_struct_pos_mapped + self.input_entity_struct_pos,
                                                          dim=-1)
            entity_lin_pos_mapped = tf.nn.l2_normalize(entity_lin_pos_mapped + self.input_entity_lin_pos, dim=-1)
            entity_img_neg_mapped = tf.nn.l2_normalize(entity_img_neg_mapped + self.input_entity_img_neg, dim=-1)
            entity_struct_neg_mapped = tf.nn.l2_normalize(entity_struct_neg_mapped + self.input_entity_struct_neg,
                                                          dim=-1)
            entity_lin_neg_mapped = tf.nn.l2_normalize(entity_lin_neg_mapped + self.input_entity_lin_neg, dim=-1)

        with tf.name_scope("rnn"):
            with tf.variable_scope("desc_model"):
                e_pos_rnn_output, e_pos_rnn_forward, e_pos_rnn_backward = self.bidirectional_rnn_model(
                    embedding_entity_desc_pos)
            with tf.variable_scope("desc_model", reuse=True):
                e_neg_rnn_output, e_neg_rnn_forward, e_neg_rnn_backward = self.bidirectional_rnn_model(
                    embedding_entity_desc_neg)
            with tf.variable_scope("desc_model", reuse=True):
                mention_rnn_output, mention_rnn_forward, mention_rnn_backward = self.bidirectional_rnn_model(
                    embedding_context)
        return mention_img_mapped,mention_lin_mapped,\
               entity_img_pos_mapped,entity_struct_pos_mapped,entity_lin_pos_mapped,\
               entity_img_neg_mapped,entity_struct_neg_mapped,entity_lin_neg_mapped, \
               e_pos_rnn_output, e_pos_rnn_forward, e_pos_rnn_backward, \
               e_neg_rnn_output, e_neg_rnn_forward, e_neg_rnn_backward, \
               mention_rnn_output, mention_rnn_forward, mention_rnn_backward

    # lstm核
    def lstm_cell(self):
        return tf.contrib.rnn.BasicLSTMCell(self.config.hidden_size, state_is_tuple=True)

    # gru核
    def gru_cell(self):
        return tf.contrib.rnn.GRUCell(self.config.hidden_size)

    # 为每一个rnn核后面加一个dropout层
    def dropout(self):
        if (self.config.rnn == 'lstm'):
            cell = self.lstm_cell()
        else:
            cell = self.gru_cell()
        return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

    def bidirectional_rnn_model(self,input):
        cells_fw = [self.dropout() for _ in range(self.config.num_layers)]
        rnn_cell_fw = tf.contrib.rnn.MultiRNNCell(cells_fw, state_is_tuple=True)
        cells_bw = [self.dropout() for _ in range(self.config.num_layers)]
        rnn_cell__bw = tf.contrib.rnn.MultiRNNCell(cells_bw, state_is_tuple=True)
        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=rnn_cell_fw, cell_bw=rnn_cell__bw, inputs=input,
                                                          dtype=tf.float32)
        state_forward = states[0][-1][-1]
        # state_forward = l2_normalize(state_forward, dim=-1)
        state_backward = states[1][-1][-1]
        # state_backward = l2_normalize(state_backward, dim=-1)
        return outputs, state_forward, state_backward

    def my_dense(self, x, nr_hidden, scope, ac_fun=tf.nn.relu):
        """
           自定义的映射层
       """
        with tf.variable_scope(scope):
            h = tf.contrib.layers.fully_connected(x, nr_hidden,activation_fn=ac_fun,
                                                  reuse=tf.AUTO_REUSE ,scope=scope)
            # , weights_regularizer= self.max_norm_regulizer
            return tf.nn.dropout(h, keep_prob = self.keep_prob)


