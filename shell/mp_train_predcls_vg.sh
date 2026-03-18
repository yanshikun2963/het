#!/bin/bash
export CUDA_VISIBLE_DEVICES="1"
export num_gpu=1
export use_multi_gpu=False
export use_obj_refine=False
export task='predcls'
export save_result=false

export REPEAT_FACTOR=0.13
export INSTANCE_DROP_RATE=1.6

export model_config="mp_vg" # relHetSGG_vg, relHetSGGp_vg
export output_dir="/home/yj/zgw/het/ckpt/${task}-mope_lxmert3-vg"

weight_files=(      
            # "model_0015000.pth"
            # "model_0020000.pth"
            # "model_0025000.pth" 
            "model_0030000.pth"
            "model_0035000.pth"
            "model_0040000.pth" 
            "model_0045000.pth" 
            "model_0050000.pth"
            "model_0055000.pth"
            "model_0060000.pth"
             )

export path_faster_rcnn='/home/yj/zgw/het/Datasets/VG/vg_faster_det.pth' # Put faster r-cnn path
mkdir ${output_dir}
cp /home/yj/zgw/het/hetsgg/modeling/roi_heads/relation_head/roi_relation_predictors.py ${output_dir}/


python tools/relation_train_net.py --config-file "configs/${model_config}.yaml" \
    SOLVER.IMS_PER_BATCH 4 \
    TEST.IMS_PER_BATCH 4     \
    OUTPUT_DIR ${output_dir} \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
    MODEL.ROI_RELATION_HEAD.REL_OBJ_MULTI_TASK_LOSS ${use_obj_refine} \
    MODEL.ROI_RELATION_HEAD.OBJECT_CLASSIFICATION_REFINE ${use_obj_refine} \
    MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING_PARAM.REPEAT_FACTOR ${REPEAT_FACTOR} \
    MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING_PARAM.INSTANCE_DROP_RATE ${INSTANCE_DROP_RATE} \
    MODEL.PRETRAINED_DETECTOR_CKPT ${path_faster_rcnn}


for weight_file in "${weight_files[@]}"
do
    echo "Running with weight file: ${weight_file}"
    
    python tools/relation_test_net.py --config-file "${output_dir}/config.yml" \
            TEST.IMS_PER_BATCH 8 \
            TEST.SAVE_RESULT ${save_result} \
            OUTPUT_DIR ${output_dir} \
            MODEL.WEIGHT "${output_dir}/${weight_file}"
done