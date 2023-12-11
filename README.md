Code for paper *Visual Chain-of-Thought Prompting for Knowledge-based Visual Reasoning*
## Overall framework
![[framework.pdf]]
## Preprocess
* Coco dataset 2014 and 2017
* Download OK-VQA and AOK-VQA dataset, following the [PICa](https://github.com/microsoft/PICa) format
* Run preprocess script (`preprocess/preprocess_aokvqa.sh` for AOK-VQA and `preprocess/preprocess_okvqa.sh`) for OK-VQA
* Make training object similarity file (`object_similarity/object_similarity_aokvqa.sh` for AOK-VQA and `object_similarity/object_similarity_okvqa.sh` for OK-VQA)
## Run experiments
* `run_aokvqa.sh` for AOK-VQA
* `run_okvqa.sh` for OK-VQA
## Main Results
| Backbone    | OK-VQA test (DA) | AOK-VQA val (DA) | AOK-VQA test (DA) |
|-------------|------------------|------------------|-------------------|
| OPT-66B     | 44.6             | 46.4             | 46.0              |
| Llama-2-70B | 54.9             | 50.5             | 54.4              |
