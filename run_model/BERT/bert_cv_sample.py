import pickle
from run_model.BERT.preprocess import load_dataset
from run_model.BERT.model import make_entity_start_model, make_entity_start_encoder
from run_model.BERT.evaluation import cv_keras_cocoscore, cv_logit_cocoscore


__author__ = 'Nuttapong Mekvipad (n.mekvipad@hotmail.com)'


# data set and BERT directory
model_path = 'biobert_large/'
ckpt_file = 'bio_bert_large_1000k.ckpt'
vocab_file = 'vocab_cased_pubmed_pmc_30k.txt'
train_path = 'dataset_swapped_9606_-26_train_train.tsv.gz'
test_path = 'dataset_swapped_9606_-26_train_test.tsv.gz'

# load train data
train_df, (x, ent1_position, ent2_position, all_len), y_train = load_dataset(train_path, model_path, vocab_file, max_seq_len=150, return_len=True)
train_df = train_df.iloc[:25000, :]
x_train = (x[:25000, :], ent1_position[:25000, :], ent2_position[:25000, :], all_len[:25000])
y_train = y_train[:25000]

# load test data
test_df, (x, ent1_position, ent2_position, all_len), y_test = load_dataset(test_path, model_path, vocab_file, max_seq_len=150, return_len=True)
test_df = test_df.iloc[:10000, :]
x_test = (x[:10000, :], ent1_position[:10000, :], ent2_position[:10000, :], all_len[:10000])
y_test = y_test[:10000]


###############################################################
#                   end-to-end BERT model                     #
###############################################################

# directory and parameters
base_params = {'bert_path': model_path,
               'ckpt_file': ckpt_file,
               'max_seq_len': 150,
               'bert_dim': 1024,
               'batch_size': 128,
               'epochs': 15,
               'cocoscore_params': {'document_weight': 0.0, 'paragraph_weight': 0.0,
                                    'sentence_weight': 1.0, 'weighting_exponent': 0.65,
                                    'score_cutoff': 0.0},
               'warn_missing_scores': False,
               'metric': 'roc_auc_score'}

params_list = [{**base_params, 'input_type': 'tokens-start-end'}]
performance = cv_keras_cocoscore(data_df=train_df, x=x_train, y=y_train, n_fold=3, model_fn=make_entity_start_model,
                                 model_params_set=params_list)

with open('end_to_end_cv_res.pickle', 'wb') as f:
    pickle.dump(performance, f)

###############################################################
#                    logistic-BERT model                      #
###############################################################

encoder_params = {'bert_path': model_path,
                  'ckpt_file': ckpt_file,
                  'max_seq_len': 150,
                  'bert_dim': 1024,
                  'batch_size': 128,
                  'input_type': 'tokens-start-end'}

logistic_params_set = [{'random_state': 0, 'max_iter':10000}]

cocoscore_params = {'cocoscore_params': {'document_weight': 0.0, 'paragraph_weight': 0.0,
                                         'sentence_weight': 1.0, 'weighting_exponent': 0.65,
                                         'score_cutoff': 0.0},
                    'warn_missing_scores': False,
                    'metric': 'roc_auc_score'}

performance = cv_logit_cocoscore(data_df=train_df, x=x_train, y=y_train, n_fold=3, encoder_params=encoder_params,
                                 logistic_params_set=logistic_params_set, cocoscore_params=cocoscore_params,
                                 model_fn=make_entity_start_encoder)

with open('logit_cv_res.pickle', 'wb') as f:
    pickle.dump(performance, f)

