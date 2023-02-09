#!/bin/bash
#SBATCH --job-name=eval_dev       # 任务名
#SBATCH --nodes=1                   # 这里不用动 多节点脚本请查官方文档
#SBATCH --ntasks=1                  # 这里不用动 多任务脚本请查官方文档
#SBATCH --cpus-per-task=4           # 要几块CPU (一般4块就够用了)
#SBATCH --mem=256GB                 # 最大内存
#SBATCH --time=12:00:00           # 运行时间上限
#SBATCH --mail-type=END             # ALL / END
#SBATCH --mail-user=yh2689@nyu.edu  # 结束之后给哪里发邮件
#SBATCH --output=./logs/eval/%x%A.out           # 正常输出写入的文件
#SBATCH --error=./logs/eval/%x%A.err            # 报错信息写入的文件
#SBATCH --gres=gpu:1                # 需要几块GPU (同时最多8块)
#SBATCH -p gpu                   # 有GPU的partition
#SBATCH --qos=gpu-8                 # To enable the use of up to 8 GPUs

nvidia-smi
nvcc --version
cd /l/users/yichen.huang/checkthat/code   # 切到程序目录

echo "START"               # 输出起始信息
source /apps/local/anaconda3/bin/activate preslav          # 调用 virtual env
# python -u inference.py  \
#     --name albef_mm_noOcr_ \
#     --model albef \
#     --no_ocr \
#     --checkpoint ../results/checkpoints/finetune_albef_mm_noOcr_/albef__batchsize32_lr0.0001_220_71_0.7534246575342466.bin \
#     --dev_set
# python -u inference.py  \
#     --name albef_pool_vis \
#     --model albef \
#     --pooling mean \
#     --image_only \
#     --checkpoint ../results/checkpoints/finetune_albef_pool_vis/albef_mean_batchsize32_lr0.0001_560_71_0.5606060606060607.bin \
#     --dev_set
# python -u inference.py  \
#     --name albef_mm_pool_ \
#     --model albef \
#     --pooling mean \
#     --checkpoint ../results/checkpoints/finetune_albef_mm_pool_/albef_mean_batchsize32_lr0.0001_570_71_0.726027397260274.bin \
#     --dev_set
# python -u inference.py  \
#     --name albef_text_noOcr_ \
#     --model albef \
#     --text_only \
#     --no_ocr \
#     --checkpoint ../results/checkpoints/finetune_albef_text_noOcr_/albef__batchsize32_lr0.0001_450_71_0.736111111111111.bin \
#     --dev_set
# python -u inference.py  \
#     --name albef_text_ \
#     --model albef \
#     --text_only \
#     --checkpoint ../results/checkpoints/finetune_albef_text_/albef__batchsize32_lr0.0001_750_71_0.7346938775510203.bin \
#     --dev_set
# python -u inference.py  \
#     --name albef_vis_ \
#     --model albef \
#     --image_only \
#     --checkpoint ../results/checkpoints/finetune_albef_vis_/albef__batchsize32_lr0.0001_530_71_0.5538461538461539.bin \
#     --dev_set
# python -u inference.py  \
#     --name albef_mm_lr1e-4_ \
#     --model albef \
#     --checkpoint ../results/checkpoints/finetune_albef_mm_lr1e-4_/albef__batchsize32_lr0.0001_510_71_0.7549668874172186.bin \
#     --dev_set
# python -u inference.py  \
#     --name blip_mm_lr1e-4_pool_ \
#     --pooling mean \
#     --checkpoint ../results/checkpoints/finetune_blip_mm_lr1e-4_pool_/blip_mean_batchsize32_lr0.0001_360_71_0.7338129496402878.bin \
#     --dev_set
python -u inference.py  \
    --name finetune_blip_mm \
    --checkpoint ../results/checkpoints/finetune_blip_mm/blip__batchsize32_lr0.0001_640_71_0.728476821192053.bin \
    --dev_set
python -u inference.py  \
    --name finetune_albef_mm_metadata_ \
    --model albef \
    --metadata \
    --checkpoint ../results/checkpoints/finetune_albef_mm_metadata_/albef__batchsize32_lr0.0001_680_71_0.7567567567567568.bin \
    --dev_set
echo "FINISH"                       # 输出起始信息
