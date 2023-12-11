BASE="/PATH/TO/VisualCOT"
engine=opt
# engine choices:
# opt, llama, bloom
# 'ada', 'babbage', 'curie', 'davinci' (gpt-3), chat (gpt-3.5-turbo)
# chat-test (for debug only)
apikey=YOUR_OPENAI_API_KEY

# parallel run 24 jobs
for i in {0..23}
do
start=$(echo 'scale=4;'$i'/24' | bc)
end=$(echo 'scale=4;('$i'+1)/24' | bc)
echo $start
echo $end
python main_aokvqa.py \
       --apikey ${apikey} \
       --output_path output/${engine}/${i} \
       --caption_type vinvl_ocr \
       --n_shot 8 \
       --n_ensemble 5 \
       --rounds 5 \
       --iterative_strategy caption \
       --engine ${engine} \
       --sg_path ${BASE}/input_text/scene_graph_text \
       --train_sim_metric answer \
       --train_sim_file "./input_text/scene_graph_text/train_object_select_answer_sim.pk" \
       --chain_of_thoughts \
       --start ${start} \
       --with_clip_verify \
       --end ${end} &
       # use --llama_path if engine == "llama"
done
