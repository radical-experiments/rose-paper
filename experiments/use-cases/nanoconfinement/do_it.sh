SEED_AL=42
echo "==============================="
echo "==============================="
echo "==============================="
echo "Doing first active learning"
python active_learning.py \
    --seed ${SEED_AL}    \
    --iter 0    \
    --pipeline_dir /eagle/RECUP/twang/rose/nanoconfinement/nanoconfinement-md/python/surrogate_samplesize/exp/pipeline_1  \
    --data_dir /eagle/RECUP/twang/rose/nanoconfinement/nanoconfinement-md/python/surrogate_samplesize/all_data_4050         \
    --new_sample_size 500

echo "==============================="
echo "==============================="
echo "==============================="
echo "Doing first training"
python train.py \
    --iter 0  \
    --instance 0 \
    --data_dir /eagle/RECUP/twang/rose/nanoconfinement/nanoconfinement-md/python/surrogate_samplesize/all_data_4050         \
    --pipeline_dir /eagle/RECUP/twang/rose/nanoconfinement/nanoconfinement-md/python/surrogate_samplesize/exp/pipeline_1  \
    --epochs 200

echo "==============================="
echo "==============================="
echo "==============================="
echo "Doing second active learning"
python active_learning.py \
    --seed ${SEED_AL}    \
    --iter 1    \
    --pipeline_dir /eagle/RECUP/twang/rose/nanoconfinement/nanoconfinement-md/python/surrogate_samplesize/exp/pipeline_1  \
    --data_dir /eagle/RECUP/twang/rose/nanoconfinement/nanoconfinement-md/python/surrogate_samplesize/all_data_4050         \
    --new_sample_size 500

echo "==============================="
echo "==============================="
echo "==============================="
echo "Doing second training"
python train.py \
    --iter 1  \
    --instance 0 \
    --data_dir /eagle/RECUP/twang/rose/nanoconfinement/nanoconfinement-md/python/surrogate_samplesize/all_data_4050         \
    --pipeline_dir /eagle/RECUP/twang/rose/nanoconfinement/nanoconfinement-md/python/surrogate_samplesize/exp/pipeline_1  \
    --epochs 200


#python active_learning.py \
#    --seed ${SEED}    \
#    --iter 2    \
#    --index_file_dir /eagle/RECUP/twang/rose/nanoconfinement/nanoconfinement-md/python/surrogate_samplesize/exp/pipeline_1  \
#    --data_dir /eagle/RECUP/twang/rose/nanoconfinement/nanoconfinement-md/python/surrogate_samplesize/all_data_4050         \
#    --new_sample_size 7
#
#python active_learning.py \
#    --seed ${SEED}    \
#    --iter 3    \
#    --index_file_dir /eagle/RECUP/twang/rose/nanoconfinement/nanoconfinement-md/python/surrogate_samplesize/exp/pipeline_1  \
#    --data_dir /eagle/RECUP/twang/rose/nanoconfinement/nanoconfinement-md/python/surrogate_samplesize/all_data_4050         \
#    --new_sample_size 7
