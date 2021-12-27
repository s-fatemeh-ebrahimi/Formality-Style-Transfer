# Formality-Style-Transfer


## 2. model scripts
**sls_rnn_fr.py** includes the APIs for training and testing the RNN-based S2S model on training dataset.

TensorFlow version is 1.12.0 used for this project.



## a. prepare data: 
The new_exp_fr folder, which include my_embedding_file and five pickle file, is  available on this link:  https://drive.google.com/drive/folders/1T-UlL24VmQOQCRmyX_3D66jbvnG2iogr?usp=sharing

Data has been prepared in new_exp_fr folder.



## b. running train stage:
Modify the parameters in sls_settings_v2_FR.py if you need.
Running sls_rnn_fr.train(). 



## c. running test stage:
Running sls_rnn_fr.test()
