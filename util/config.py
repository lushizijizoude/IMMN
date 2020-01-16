# coding: utf-8

"""
对模型进行相关配置的文件
"""
import os

class Config_Util(object):

    #project path
    base_path = "/home3/jason/MKG/"

    save_dir = base_path + "save/"
    tensorboard_dir = base_path + "save/tensorboard"
    pre_train_embed_path = base_path + "data/Wiki4MNED/embeddings/linguistic/word2vec/model"
    input_dir = base_path + "data/Wiki4MNED/candidate/"
    vocab_dir = base_path + "data/Wiki4MNED/embeddings/linguistic/word2vec/"

    #embedding path
    muti_struct_embedding_path = base_path + "data/Wiki4MNED/embeddings/mtrl/mtrl_struct_100.pkl"
    muti_entity_img_embedding_path = base_path + "data/Wiki4MNED/embeddings/mtrl/mtrl_img_100.pkl"
    transE_struct_embedding_path = base_path + "data/Wiki4MNED/embeddings/transE/transE_100.pkl"
    mention_img_embedding_path = base_path + "data/Wiki4MNED/embeddings/post/vgg19_mean_100.pkl"



    margin = 0.3
    num_epochs = 10
    batch_size = 128
    val_batch_size = 1024
    save_per_batch = 1000
    print_per_batch = 1000
    require_improvement = 400000000
    rnn = "lstm"  # lstm or gru
    num_classes = 2
    num_layers = 1
    hidden_size = 100
    seq_length = 64
    attention_size = 64
    embedding_dim = 100
    pos_embd_dim = 5
    pos_size = 100
    is_pre_train_embed = True
    dropout_keep_prob = 0.8
    learning_rate = 1e-3



