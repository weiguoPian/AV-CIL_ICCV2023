dataset=$1;

if [ $dataset = AVE ];then
    num_classes=28
    class_num_per_step=7
    num_workers=0
    memory_size=340
    max_epoches=200
    lam_I=0.5
    lam_C=1.0
elif [ $dataset = ksounds ];then
    num_classes=30
    class_num_per_step=6
    num_workers=4
    memory_size=500
    max_epoches=200
    lam_I=0.1
    lam_C=1.0
elif [ $dataset = VGGSound_100 ];then
    num_classes=100
    class_num_per_step=10
    num_workers=4
    memory_size=1500
    max_epoches=200
    lam_I=0.1
    lam_C=1.0
else
    echo "dataset must be \"AVE\", \"ksound\", or \"VGGSound_100\".";
    exit;
fi

cd ours

CUDA_VISIBLE_DEVICES=3 nohup python -u train_incremental_ours.py \
                                --dataset $dataset \
                                --num_classes $num_classes \
                                --class_num_per_step $class_num_per_step \
                                --max_epoches $max_epoches \
                                --num_workers $num_workers \
                                --memory_size $memory_size \
                                --instance_contrastive \
                                --class_contrastive \
                                --attn_score_distil \
                                --instance_contrastive_temperature 0.05 \
                                --class_contrastive_temperature 0.05 \
                                --lam 0.5 \
                                --lam_I $lam_I \
                                --lam_C $lam_C \
                                --lr 1e-3 \
                                --lr_decay False \
                                --milestones 100 \
                                --weight_decay 1e-4 \
                                --train_batch_size 256 \
                                --infer_batch_size 128 \
                                --exemplar_batch_size 128 > nohup_ours.log 2>&1 &


