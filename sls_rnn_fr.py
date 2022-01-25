from sls_rnn import prepare_data,train_onehotkey_with_multi_datasets,generate_copy
from sls_settings_v2_FR import *
from sls_settings_v2_FR import train_parameters,generate_parameters,embedding_path
import os
import time
import pickle
from evaluate_all import evaluate

def get_timestamp():
    return time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())

def train_rnn_combined():
    # embedding_file= '/content/gdrive/MyDrive/en.wiki.bpe.vs3000.d300.w2v.txt'
    # add_inf_data = '/content/gdrive/MyDrive/data/Entertainment_Music/train/informal'
    # add_fm_data = '/content/gdrive/MyDrive/data/Entertainment_Music/train/formal'
    # add_data_big = prepare_data(add_inf_data,
    #                             add_fm_data,
    #                             embedding_file=embedding_path,
    #                             to_lower_case=False)
    # pickle.dump(add_data_big, open(add_data_big_pkl_path, 'wb'), protocol=True)
    para = train_parameters(arch_type='rnn_combined')
    if not os.path.exists(para.save_dir):
        os.makedirs(para.save_dir)
    best_step_num, best_val_bleu = train_onehotkey_with_multi_datasets(parameters=para)
    print(str(best_step_num) + ',' + str(best_val_bleu))

def train_rnn():
    para = train_parameters(arch_type='rnn')
    if not os.path.exists(para.save_dir):
        os.makedirs(para.save_dir)
    best_step_num, best_val_bleu = train_onehotkey_with_multi_datasets(parameters=para)
    print(str(best_step_num) + ',' + str(best_val_bleu))

def test():
    best_step_num=3600
    print("start testing1")
    gen_para = generate_parameters()
    gen_para.decode_alpha = 0.8
    gen_para.beam_size = 30
    gen_para.model_path = './new_exp_fr/model_doamin_combined/' + 'best.' + str(best_step_num) + '.model'
    generate_copy(para=gen_para, copy=True, save_bpe_result=True)
    test_bleu = evaluate()
    print("test_bleu:", str(test_bleu))

if __name__=='__main__':
    train_rnn()
    # train_rnn_combined()
    #test()





