CUDA_VISIBLE_DEVICES=0,1,2,3 python train_MTA.py --savename resnet18_MTA_stage1 \
    --arch_teacher resnet18 --source saved_models/resnet18_CE.pth.tar \
    --logname resnet18_MTA_stage1 

CUDA_VISIBLE_DEVICES=0,1,2,3 python train_MTA.py --savename resnet18_MTA_stage2 \
    --arch_teacher resnet18 --source saved_models/resnet18_CE.pth.tar \
    --logname resnet18_MTA_stage2 --pretrained saved_models/resnet18_MTA_stage1.pth.tar\
    --batch_size 36 --attack_decay_iter 3000 --eps_c 1200 --max_iteration 50000

CUDA_VISIBLE_DEVICES=0,1,2,3 python train_MTA.py --savename resnet18_MTA_stage3 \
    --arch_teacher resnet18 --source saved_models/resnet18_CE.pth.tar \
    --logname resnet18_MTA_stage3 --pretrained saved_models/resnet18_MTA_stage2.pth.tar\
    --batch_size 24 --attack_decay_iter 3000 --eps_c 1200 --max_iteration 50000
