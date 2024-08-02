CUDA_VISIBLE_DEVICES="0" python ./main.py --mode zsl \
                                   --in_file_tst_dialog ./data/fs_eval_t1.wst.dev \
                                   --in_file_fashion ./data/mdata.wst.txt.2023.08.23 \
                                   --in_dir_img_feats ./data/img_feats \
                                   --subWordEmb_path ./sstm_v0p5_deploy/sstm_v4p49_np_n36134_d128.dat \
                                   --model_path ./model \
                                   --model_file gAIa-500.pt \
                                   --req_net_type memn2n \
                                   --eval_net_type tf \
                                   --mem_size 32 \
                                   --key_size 300 \
                                   --hops 3 \
                                   --eval_node [600,4000,4000] \
                                   --tf_nhead 4 \
                                   --tf_ff_dim 4096 \
                                   --tf_num_layers 4 \
                                   --batch_size 100 \
                                   --use_multimodal True \

