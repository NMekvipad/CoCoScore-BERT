import pickle
from run_model.BERT.preprocess import load_dataset
from run_model.BERT.model import make_entity_start_model, make_entity_start_encoder
from run_model.BERT.evaluation import train_model, eval_model, train_logit_bert, eval_logit_bert


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
model_params = {'bert_path': model_path,
                'ckpt_file': ckpt_file,
                'max_seq_len': 150,
                'bert_dim': 1024,
                'batch_size': 128,
                'epochs': 15}

cocoscore_params = {'document_weight': 0.0, 'paragraph_weight': 0.0,
                    'sentence_weight': 1.0, 'weighting_exponent': 0.65,
                    'score_cutoff': 0.0}


model = train_model(x=x_train, y=y_train, model_params=model_params, model_fn=make_entity_start_model,
                    input_type='tokens-start-end', checkpoint_file='./checkpoint_file/training_2nd-0008.ckpt',
                    create_checkpoint=False)

score, res_df = eval_model(test_df, x_test, model, cocoscore_params, input_type='tokens-start-end',
                           warn_missing_scores=False, metric='roc_auc_score', return_cocoscores=False, baseline=False,
                           return_dataframe=True)

with open('end_to_end_res.pickle', 'wb') as f:
    pickle.dump(score, f)

###############################################################
#                    logistic-BERT model                      #
###############################################################

encoder_params = {'bert_path': model_path,
                  'ckpt_file': ckpt_file,
                  'max_seq_len': 150,
                  'bert_dim': 1024,
                  'batch_size': 128,
                  'input_type': 'tokens-start-end'}

logistic_params = {'random_state': 0, 'max_iter':10000}

cocoscore_params = {'document_weight': 0.0, 'paragraph_weight': 0.0,
                    'sentence_weight': 1.0, 'weighting_exponent': 0.65,
                    'score_cutoff': 0.0}

model = train_logit_bert(x_train, y_train, encoder_params, logistic_params, encoder_fn=make_entity_start_encoder)
score = eval_logit_bert(test_df, x_test, model, cocoscore_params, encoder_params,
                        encoder_fn=make_entity_start_encoder,
                        warn_missing_scores=False, metric='roc_auc_score',
                        return_cocoscores=False, baseline=False)

with open('logit_res.pickle', 'wb') as f:
    pickle.dump(score, f)


