export CUDA_VISIBLE_DEVICES="0"
export num_gpu=1
export use_multi_gpu=False
export use_obj_refine=False
export task='predcls'

export REPEAT_FACTOR=0.13
export INSTANCE_DROP_RATE=1.6

export model_config="mp_vg" # relHetSGG_vg, relHetSGGp_vg
export output_dir="/home/yj/zgw/het/ckpt/${task}-het_prompt_mp-vg_1"

export path_faster_rcnn='/root/het/Datasets/VG/vg_faster_det.pth' # Put faster r-cnn path
cp /home/yj/zgw/het/hetsgg/modeling/roi_heads/relation_head/roi_relation_predictors.py ${output_dir}/

if $use_multi_gpu;then
    python -m torch.distributed.launch  --master_port 10029 --nproc_per_node=${num_gpu} tools/relation_train_net.py \
        --config-file "configs/${model_config}.yaml" \
        SOLVER.IMS_PER_BATCH 36 \
        TEST.IMS_PER_BATCH 24 \
        OUTPUT_DIR ${output_dir} \
        MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
        MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
        MODEL.ROI_RELATION_HEAD.REL_OBJ_MULTI_TASK_LOSS ${use_obj_refine} \
        MODEL.ROI_RELATION_HEAD.OBJECT_CLASSIFICATION_REFINE ${use_obj_refine} \
        MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING_PARAM.REPEAT_FACTOR ${REPEAT_FACTOR} \
        MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING_PARAM.INSTANCE_DROP_RATE ${INSTANCE_DROP_RATE} \
        MODEL.PRETRAINED_DETECTOR_CKPT ${path_faster_rcnn}
else
    python tools/relation_train_net.py --config-file "configs/${model_config}.yaml" \
        SOLVER.IMS_PER_BATCH 4 \
        TEST.IMS_PER_BATCH 1     \
        OUTPUT_DIR ${output_dir} \
        MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
        MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
        MODEL.ROI_RELATION_HEAD.REL_OBJ_MULTI_TASK_LOSS ${use_obj_refine} \
        MODEL.ROI_RELATION_HEAD.OBJECT_CLASSIFICATION_REFINE ${use_obj_refine} \
        MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING_PARAM.REPEAT_FACTOR ${REPEAT_FACTOR} \
        MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING_PARAM.INSTANCE_DROP_RATE ${INSTANCE_DROP_RATE} \
        MODEL.PRETRAINED_DETECTOR_CKPT ${path_faster_rcnn}
fi
