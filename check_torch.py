import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name())



# python preprocess.py --train_src data/src-train-tokenized.txt --valid_src data/src-valid-tokenized.txt --train_knl data/knl-train-tokenized.txt --valid_knl data/knl-valid-tokenized.txt --train_tgt data/tgt-train-tokenized.txt --valid_tgt data/tgt-valid-tokenized.txt --save_data data/cmu_movie -dynamic_dict -share_vocab -src_seq_length_trunc 50 -tgt_seq_length_trunc 50 -knl_seq_length_trunc 200 -src_seq_length 150 -knl_seq_length 800
#
#
# python preprocess.py --train_src data/src-valid-tokenized.txt --valid_src data/src-valid-tokenized.txt --train_knl data/knl-valid-tokenized.txt --valid_knl data/knl-valid-tokenized.txt --train_tgt data/tgt-valid-tokenized.txt --valid_tgt data/tgt-valid-tokenized.txt --save_data data/cmu_movie -dynamic_dict -share_vocab -src_seq_length_trunc 50 -tgt_seq_length_trunc 50 -knl_seq_length_trunc 200 -src_seq_length 150 -knl_seq_length 800
#
# python train.py -config config/config-transformer-base-1GPU.yml