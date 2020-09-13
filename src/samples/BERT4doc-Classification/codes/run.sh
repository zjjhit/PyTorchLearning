export CUDA_VISIBLE_DEVICES=0,1,2,3
python finetuning/run_classifier.py \
  --task_name ag \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir ../data/ag_news_csv/ \
  --vocab_file ../data/bert-base-cased/vocab.txt \
  --bert_config_file ../data/bert-base-cased/bert_config.json \
  --init_checkpoint ../data/bert-base-cased/pytorch_model.bin \
  --max_seq_length 512 \
  --train_batch_size 24 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir ../data/ag_out/ \
  --seed 42

:<<EOF
python finetuning/convert_tf_checkpoint_to_pytorch.py \
  --tf_checkpoint_path ../data/bert-base-cased/uncased_L-12_H-768_A-12/bert_model.ckpt \
  --bert_config_file ../data/bert-base-cased/uncased_L-12_H-768_A-12/bert_config.json \
  --pytorch_dump_path ../data/bert-base-cased/uncased_L-12_H-768_A-12/pytorch_model.bin
EOF
