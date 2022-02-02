# Formality-Style-Transfer


## 2. model scripts
**sls_rnn_fr.py** includes the APIs for training and testing the RNN-based S2S model on training dataset.

TensorFlow version is 1.12.0 used for this project.



## a. prepare data: 
The new_exp_fr folder, should include embedding_file and five pickle file to start training.


## b. running train stage:
Modify the parameters in sls_settings_v2_FR.py if you need.
Running sls_rnn_fr.train(). 



## c. running test stage:
Running sls_rnn_fr.test()
