
export MODEL_DIR=${HOME}/nmt_data/models/medium100000LSTM
export VOCAB_SOURCE=${HOME}/nmt_data/oracledata_reverse/train/vocab.sourcesall.txt
export VOCAB_TARGET=${HOME}/nmt_data/oracledata_reverse/train/vocab.targetsall.txt
export TRAIN_SOURCES=${HOME}/nmt_data/oracledata_reverse/train/sourcesall.txt
export TRAIN_TARGETS=${HOME}/nmt_data/oracledata_reverse/train/targetsall.txt
export DEV_SOURCES=${HOME}/nmt_data/oracledata_reverse/dev/sourcesall200.txt
export DEV_TARGETS=${HOME}/nmt_data/oracledata_reverse/dev/targetsall200.txt

export DEV_TARGETS_REF=${HOME}/nmt_data/oracledata_reverse/dev/targetsall200.txt
export TRAIN_STEPS=100000
mkdir -p $MODEL_DIR

python -m bin.train \
  --config_paths="
      ../seq2seq/example_configs/nmt_large.yml,
      ../seq2seq/example_configs/train_seq2seq.yml,
      ../seq2seq/example_configs/text_metrics_bpe.yml" \
  --model_params "
      vocab_source: $VOCAB_SOURCE
      vocab_target: $VOCAB_TARGET" \
  --input_pipeline_train "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - $TRAIN_SOURCES
      target_files:
        - $TRAIN_TARGETS" \
  --input_pipeline_dev "
    class: ParallelTextInputPipeline
    params:
       source_files:
        - $DEV_SOURCES
       target_files:
        - $DEV_TARGETS" \
  --batch_size 32 \
  --train_steps 100000 \
  --output_dir $MODEL_DIR

tensorboard --logdir $MODEL_DIR

export PRED_DIR=${MODEL_DIR}/pred
mkdir -p ${PRED_DIR}

python -m bin.infer \
  --tasks "
    - class: DecodeText
    - class: DumpBeams
      params:
        file: ${PRED_DIR}/beams.npz" \
  --model_dir $MODEL_DIR \
  --model_params "
    inference.beam_search.beam_width: 10"\
  --input_pipeline "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - $DEV_SOURCES" \
  > ${PRED_DIR}/predictionsall100000LSTM.txt


../seq2seq/bin/tools/multi-bleu.perl ${DEV_TARGETS_REF} < ${PRED_DIR}/predictionsall100000LSTM.txt