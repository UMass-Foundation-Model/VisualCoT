BASE="/PATH/TO/VisualCOT"
engine=opt
apikey=YOUR_OPENAI_API_KEY
for i in {0..79}
do
start=$(echo 'scale=4;'$i'/80' | bc)
end=$(echo 'scale=4;('$i'+1)/80' | bc)
echo $start
echo $end
python main_okvqa.py \
       --apikey ${apikey} \
       --output_path output/okvqa_llama2/${i} \
       --caption_type vinvl_ocr \
       --n_shot 8 \
       --n_ensemble 5 \
       --rounds 5 \
       --engine ${engine} \
       --iterative_strategy caption \
       --sg_path ${BASE}/input_text/scene_graph_text \
       --train_sim_metric answer \
       --train_sim_file "./input_text/scene_graph_text/train_object_select_okvqa.pk" \
       --with_clip_verify \
       --start ${start} \
       --with_ok_context \
       --end ${end}  &
done
