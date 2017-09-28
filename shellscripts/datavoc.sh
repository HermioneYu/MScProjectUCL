set -e

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"

DATA_TYPE=${DATA_TYPE:-copy}
echo "Using type=${DATA_TYPE}. To change this set DATA_TYPE to 'copy' or 'reverse'"

OUTPUT_DIR=${OUTPUT_DIR:-$HOME/nmt_data/oracledata_${DATA_TYPE}}
echo "Writing to ${OUTPUT_DIR}. To change this, set the OUTPUT_DIR environment variable."

OUTPUT_DIR_TRAIN="${OUTPUT_DIR}/train"
OUTPUT_DIR_DEV="${OUTPUT_DIR}/dev"
OUTPUT_DIR_TEST="${OUTPUT_DIR}/test"



# Create Vocabulary

${BASE_DIR}/Hermione/seq2seq/bin/tools/generate_vocab.py \
  < ${OUTPUT_DIR_TRAIN}/sourcesall.txt \
  > ${OUTPUT_DIR_TRAIN}/vocab.sourcesall.txt
echo "Wrote ${OUTPUT_DIR_TRAIN}/vocab.sourcesall.txt"

${BASE_DIR}/Hermione/seq2seq/bin/tools/generate_vocab.py \
  < ${OUTPUT_DIR_TRAIN}/targetsall.txt \
  > ${OUTPUT_DIR_TRAIN}/vocab.targetsall.txt
echo "Wrote ${OUTPUT_DIR_TRAIN}/vocab.targetsall.txt"

# Optionally encode data with google/sentencepice
# Useful for testing
if [ "$SENTENCEPIECE" = true ]; then
  spm_train \
    --input=${OUTPUT_DIR_TRAIN}/sourcesall.txt,${OUTPUT_DIR_TRAIN}/targetsall.txt \
    --model_prefix=${OUTPUT_DIR}/bpe \
    --vocab_size=20 \
    --model_type=bpe
  for dir in ${OUTPUT_DIR_TRAIN} ${OUTPUT_DIR_DEV} ${OUTPUT_DIR_TEST}; do
    spm_encode --model=${OUTPUT_DIR}/bpe.model --output_format=piece \
      < ${dir}/sourcesall.txt \
      > ${dir}/sources.bpe.txt
    spm_encode --model=${OUTPUT_DIR}/bpe.model --output_format=piece \
      < ${dir}/targetsall.txt \
      > ${dir}/targets.bpe.txt
  done
fi