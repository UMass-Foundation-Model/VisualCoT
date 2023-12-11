BASE=/PATH/TO/VisualCOT
engine=opt
python object_engineering_okvqa.py \
       --apikey <your-openai-api-key> \
       --output_path output \
       --caption_type vinvl_sg \
       --n_shot 8 \
       --iterative_strategy caption \
       --engine ${engine} \
       --sg_path ${BASE}/input_text/scene_graph_text \
       --with_six_gpus
