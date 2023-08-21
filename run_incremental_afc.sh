dataset=$1;
modality=$2;
classify=$3;


if [ $classify != NME ] && [ $classify != LSC ];then
    echo "classify must be \"NME\" or \"LSC\".";
    exit;
fi

if [ $dataset = AVE ];then
    num_classes=28
    class_num_per_step=7
    num_workers=0
    memory_size=340
    gen_exem_batch_size=256
    if [ $modality = audio ];then
        max_epoches=300
    elif [ $modality = visual ];then
        max_epoches=200
    elif [ $modality = audio-visual ];then
        max_epoches=150
    else
        echo "modality must be \"audio\", \"visual\", or \"audio-visual\".";
        exit;
    fi
elif [ $dataset = ksounds ];then
    num_classes=30
    class_num_per_step=6
    memory_size=500
    gen_exem_batch_size=64
    if [ $modality = audio ];then
        num_workers=0
        max_epoches=300
    elif [ $modality = visual ];then
        num_workers=4
        max_epoches=100
    elif [ $modality = audio-visual ];then
        num_workers=4
        max_epoches=100
    else
        echo "modality must be \"audio\", \"visual\", or \"audio-visual\".";
        exit;
    fi
elif [ $dataset = VGGSound_100 ];then
    num_classes=100
    class_num_per_step=10
    memory_size=1500
    gen_exem_batch_size=256
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
        echo "modality must be \"audio\", \"visual\", or \"audio-visual\".";
        exit;
    fi
else
    echo "dataset must be \"AVE\", \"ksound\", or \"VGGSound_100\".";
    exit;
fi

cd AFC

CUDA_VISIBLE_DEVICES=3 nohup python -u train_incremental_afc.py \
                                --dataset $dataset \
                                --num_classes $num_classes \
                                --class_num_per_step $class_num_per_step \
                                --modality $modality \
                                --classify $classify \
                                --max_epoches $max_epoches \
                                --num_workers $num_workers \
                                --memory_size $memory_size \
                                --lr 1e-3 \
                                --lr_decay False \
                                --milestones 100 \
                                --weight_decay 1e-4 \
                                --lam_disc 0.5 \
                                --train_batch_size 256 \
                                --infer_batch_size 128 \
                                --gen_exem_batch_size $gen_exem_batch_size > nohup_afc.log 2>&1 &


