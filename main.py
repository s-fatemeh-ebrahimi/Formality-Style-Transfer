import nltk
# nltk.download('punkt')

import sls_rnn

sls_rnn.preprocess()
sls_rnn.gen_val_data_for_bleu()
