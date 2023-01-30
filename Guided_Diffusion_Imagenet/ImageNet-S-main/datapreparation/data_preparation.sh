SOURCE=$1
DEST=$2
MODE=$3
COPY=$4

if [ ${COPY} == "true" ];
then
python datapreparation_train.py --imagenet-dir ${SOURCE} --save-dir ${DEST} --mode ${MODE} --copy
else
python datapreparation_train.py --imagenet-dir ${SOURCE} --save-dir ${DEST} --mode ${MODE}
fi
python datapreparation_val.py --imagenet-dir ${SOURCE} --save-dir ${DEST} --mode ${MODE}
python datapreparation_train_semi.py --imagenet-dir ${SOURCE} --save-dir ${DEST} --mode ${MODE}
bash datapreparation_anno.sh ${DEST} ${MODE}
