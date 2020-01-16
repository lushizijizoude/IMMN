
import os
from model.OurModel import OurModel
from util.config import Config_Util
from util.data_util import DataUtil
from trainer.MNEDTrainer import MNEDTrainer

if __name__ == "__main__":
    config_util = Config_Util()
    config_util.input_dir = "/home3/jason/KG/data/describe_data/Wikilinks/"
    config_util.pre_train_embed_path = config_util.base_path + "data/Wikilinks/embeddings/linguistic/word2vec/model"
    config_util.input_dir = config_util.base_path + "data/Wikilinks/candidate/"
    config_util.vocab_dir = config_util.base_path + "data/Wikilinks/embeddings/linguistic/word2vec/"
    config_util.transE_struct_embedding_path = config_util.base_path + "data/Wikilinks/embeddings/transE/transE_100.pkl"
    config_util.save_per_batch = 30000
    config_util.print_per_batch = 30000

    config_util.muti_struct_embedding_path = ""
    config_util.muti_entity_img_embedding_path = ""
    config_util.mention_img_embedding_path = ""

    data_helper = DataUtil(config_util)
    # data_helper.load_word_embedding()
    data_helper.load_embeddings()
    config_util.margin = 0.3
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    model = OurModel(config_util)
    model.use_interactive_attention = True
    model.use_self_attention = True
    model.use_mention_img_embedding = False
    model.use_entity_img_embedding = False
    model.use_tranE_struct_embedding = True
    model.use_fertures = True
    model.use_mention_attention = True


    model.model_name = "OurModel_Wikilinks"
    model.init_word_embedding(data_helper.word_embedding)
    model.build()
    section_save_path = config_util.save_dir + "pkl_files/Wikilinks/"
    entity_linking = MNEDTrainer(data_helper, model, section_save_path)
    entity_linking.show_attention = False
    entity_linking.train()
    entity_linking.group_test()