
import os
from model.OurModel import OurModel
from util.config import Config_Util
from util.data_util import DataUtil
from trainer.MNEDTrainer import MNEDTrainer

if __name__ == "__main__":
    config_util = Config_Util()
    data_helper = DataUtil(config_util)
    # data_helper.load_word_embedding()
    data_helper.load_embeddings()
    config_util.margin = 0.3
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    model = OurModel(config_util)
    model.use_interactive_attention = True
    model.use_self_attention = True
    model.use_mention_img_embedding = True
    model.use_entity_img_embedding = True
    model.use_tranE_struct_embedding = False
    model.use_fertures = True
    model.use_mention_attention = True


    model.model_name = "OurModel_Wiki4MNED"
    model.init_word_embedding(data_helper.word_embedding)
    model.build()
    section_save_path = config_util.save_dir + "pkl_files/Wiki4MNED/"
    entity_linking = MNEDTrainer(data_helper, model, section_save_path)
    entity_linking.show_attention = False
    entity_linking.train()
    entity_linking.group_test()