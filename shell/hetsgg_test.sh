export CUDA_VISIBLE_DEVICES="0"
export num_gpu=1
export use_multi_gpu=false
export task='predcls'

# export test_list=('0045000') # checkpoint

export save_result=False
export output_dir="/root/autodl-tmp/ckpt/predcls-het_prompt_mp-vg_1" # Please input the checkpoint directory


python tools/relation_test_net.py --config-file "${output_dir}/config.yml"  \
            TEST.IMS_PER_BATCH 8 \
            TEST.SAVE_RESULT ${save_result} \
            OUTPUT_DIR ${output_dir} \
            MODEL.WEIGHT "${output_dir}/model_0050000.pth"