dataset=$1;
modality=$2;

if [ $dataset = AVE ];then
    num_classes=28
    class_num_per_step=7
    num_workers=0
    if [ $modality = audio ];then
        max_epoches=300
    elif [ $modality = visual ];then
        max_epoches=200
    elif [ $modality = audio-visual ];then
        max_epoches=200
    else
        echo "modality must be \"audio\", \"visual\", or \"audio-visual\"";
        exit;
    fi
elif [ $dataset = ksounds ];then
    num_classes=30
    class_num_per_step=6
    if [ $modality = audio ];then
        num_workers=0
        max_epoches=200
    elif [ $modality = visual ];then
        num_workers=4
        max_epoches=200
    elif [ $modality = audio-visual ];then
        num_workers=4
        max_epoches=100
    else
        echo "modality must be \"audio\", \"visual\", or \"audio-visual\"";
        exit;
    fi
elif [ $dataset = VGGSound_100 ];then
    num_classes=100
    class_num_per_step=10
    if [ $modality = audio ];then
        num_workers=0
        max_epoches=300
    elif [ $modality = visual ];then
        num_workers=4
        max_epoches=200
    elif [ $modality = audio-visual ];then
        num_workers=4
        max_epoches=200
    else
        echo "modality must be \"audio\", \"visual\", or \"audio-visual\"";
        exit;
    fi
else
    echo "dataset must be \"AVE\", \"ksounds\", or \"VGGSound_100\".";
    exit;
fi

cd LwF

CUDA_VISIBLE_DEVICES=3 nohup python -u train_incremental_lwf.py \
                                --dataset $dataset \
                                --num_classes $num_classes \
                                --class_num_per_step $class_num_per_step \
                                --modality $modality \
                                --max_epoches $max_epoches \
                                --num_workers $num_workers \
                                --lr 1e-3 \
                                --lr_decay False \
                                --milestones 100 \
                                --weight_decay 1e-4 \
                                --train_batch_size 256 \
                                --infer_batch_size 128 > nohup_lwf.log 2>&1 &



             