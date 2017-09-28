export MODEL_DIR=${HOME}/nmt_data/models/medium100000
export DEV_SOURCES=${HOME}/nmt_data/oracledata_reverse/dev/sourcesall200.txt
export DEV_TARGETS_REF=${HOME}/nmt_data/oracledata_reverse/dev/targetsall200.txt
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
    inference.beam_search.beam_width: 10" \
  --input_pipeline "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - $DEV_SOURCES" \
  > ${PRED_DIR}/predictionssmldata200large1.txt


../seq2seq/bin/tools/multi-bleu.perl ${DEV_TARGETS_REF} < ${PRED_DIR}/predictionssmldata200large1.txt