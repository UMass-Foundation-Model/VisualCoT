BASE=/PATH/TO/VisualCOT
COCO17PATH=/PATH/TO/coco17
COCO14PATH=/PATH/TO/coco14

# reorganize train / val split
python reorganize_captions.py \
            --coco14root ${COCO14PATH} \
            --coco17root ${COCO17PATH} \
            --caption14train ${BASE}/coco_annotations/captions_train2014.json \
            --caption14val ${BASE}/coco_annotations/captions_val2014.json \
            --caption17train ${BASE}/coco_annotations/captions_train2017.json \
            --caption17val ${BASE}/coco_annotations/captions_val2017.json
for SPLIT in "train" "val"
do
# make line2sample
python make_line2sample.py \
           --input ${BASE}/coco_annotations/aokvqa_v1p0_${SPLIT}.json \
           --output ${BASE}/coco_clip_new/aokvqa_qa_line2sample_idx_${SPLIT}2017.json
# make clip features
python make_clip_features.py \
           --questions ${BASE}/coco_annotations/aokvqa_v1p0_${SPLIT}.json \
           --images ${COCO17PATH}/${SPLIT}2017/ \
           --ifeatures ${BASE}/coco_clip_new/coco_clip_vitb16_${SPLIT}2017_aokvqa_convertedidx_image.npy \
           --qfeatures ${BASE}/coco_clip_new/coco_clip_vitb16_${SPLIT}2017_aokvqa_question.npy
done
