docker run -it --gpus='"device=0,1"' -v /data/wufan/algos/data/transformers_data:/data transformers:v1.0.2 bash -c \
  "python3 ./examples/pytorch/text-classification/run_glue.py\
  --model_name_or_path /data/model/bert-base-uncased \
  --train_file /data/nlpcc2014/nlpcc2014Train.json \
  --validation /data/nlpcc2014/nlpcc2014Val.json \
  --test_file /data/nlpcc2014/nlpcc2014Test.json \
  --do_train  \
  --do_eval  \
  --do_predict  \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5  \
  --num_train_epochs 1 \
  --output_dir /data/nlpcc2014/workdir/ \
  --overwrite_output_dir"

docker run -it --gpus='"device=2,3"' -v /data/wufan/algos/data/transformers_data:/data transformers:v1.0.2 bash -c \
  "python3 ./examples/pytorch/text-classification/run_glue.py\
  --model_name_or_path /data/model/albert-tiny-chinese \
  --train_file /data/nlpcc2014/nlpcc2014Train.json \
  --validation /data/nlpcc2014/nlpcc2014Val.json \
  --test_file /data/nlpcc2014/nlpcc2014Test.json \
  --do_train  \
  --do_eval  \
  --do_predict  \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5  \
  --num_train_epochs 1 \
  --output_dir /data/nlpcc2014/workdir/albert-tiny-chinese \
  --overwrite_output_dir"




docker run -it --gpus='"device=2,3"' -v /data/wufan/algos/data/transformers_data:/data transformers:v1.0.2 bash -c \
  "python3 ./examples/pytorch/token-classification/run_ner.py \
  --model_name_or_path /data/model/albert-tiny-chinese \
  --train_file /data/cluener/CLUENER2020_train.json \
  --validation_file /data/cluener/CLUENER2020_dev.json \
  --test_file /data/cluener/CLUENER2020_test.json \
  --output_dir /data/cluener/workdir/albert-tiny-chinese/ \
  --do_train  \
  --do_eval \
  --do_predict  \
  --num_train_epochs 1 \
  --overwrite_output_dir"


docker run -it --gpus='"device=0,1"' -v /data/wufan/algos/data/transformers_data:/data transformers:v1.0.2 bash -c \
  "python3 ./examples/pytorch/token-classification/run_ner.py \
  --model_name_or_path /data/model/bert-base-uncased \
  --train_file /data/cluener/CLUENER2020_train.json \
  --validation_file /data/cluener/CLUENER2020_dev.json \
  --test_file /data/cluener/CLUENER2020_test.json \
  --output_dir /data/cluener/workdir/bert-base-uncased \
  --do_train  \
  --do_eval \
  --do_predict  \
  --num_train_epochs 1 \
  --overwrite_output_dir"