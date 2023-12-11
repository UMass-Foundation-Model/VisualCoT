import os
import csv
import argparse
import numpy as np
import multiprocessing
import time
import json
import time
import torch
import random
import openai
from tqdm import tqdm
from transformers import GPT2Tokenizer
import pdb
import pickle
import glob
from transformers import CLIPProcessor, CLIPModel
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image
import utils_api

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_random_seed(seed=0)

def parse_sentence(raw_result_list):
    output_list = []
    for raw_result in raw_result_list:
        if isinstance(raw_result, str):
            raw_result = raw_result.strip(" ")
            tmp_result_list = raw_result.split(",")
            tmp_output_list = []
            for tmp_result in tmp_result_list:
                tmp_output_list +=tmp_result.split(" and ")
            output_list += tmp_output_list
        elif isinstance(raw_result, list):
            raw_ouput = parse_sentence(raw_result)
            output_list +=raw_ouput
    output_list = [ele.strip() for ele in output_list]
    output_list = [ele for ele in output_list if len(ele)>0]
    return output_list

def parge_obj_name(raw_result_list):
    output_list = []
    for raw_result in raw_result_list:
        if isinstance(raw_result, str):
            raw_result = raw_result.strip(" ")
            tmp_result_list = raw_result.split(",")
            tmp_output_list = []
            for tmp_result in tmp_result_list:
                tmp_output_list +=tmp_result.split(" and ")
            output_list += tmp_output_list
        elif isinstance(raw_result, list):
            raw_ouput = parse_sentence(raw_result)
            output_list +=raw_ouput
    output_list = [ele[2:] if ele.lower().startswith("a ") else ele for ele in output_list]
    output_list = [ele[3:] if ele.lower().startswith("an ") else ele for ele in output_list]
    output_list = [ele[4:] if ele.lower().startswith("the ") else ele for ele in output_list]
    output_list = [ele.strip() for ele in output_list]
    output_list = [ele for ele in output_list if len(ele)>0]
    return output_list


## initial cleaning for reference QA results; Please use vqav2 eval script for the final number
def process_answer(answer):
    answer = answer.replace('.', '').replace(',', '').lower()
    to_be_removed = {'a', 'an', 'the', 'to', ''}
    answer_list = answer.split(' ')
    answer_list = [item for item in answer_list if item not in to_be_removed]
    return ' '.join(answer_list)

def bounding_box_matching(box1, box2, thres=0.5):
    ax1, ay1, ax2, ay2 = box1
    bx1, by1, bx2, by2 = box2
    if ax1 >= bx2 or ax2 <= bx1 or ay1 >= by2 or ay2 <= by1:
        return 0
    intersection = (min(ax2, bx2) - max(ax1, bx1)) * (min(ay2, by2) - max(ay1, by1))
    and_set = (ax2-ax1) * (ay2-ay1) + (bx2-bx1) * (by2-by1) - intersection
    return (intersection / and_set)

class VisualCOT_AOKVQA:
    def __init__(self, args, apikey_list):
        self.args = args
        self.chain_of_thoughts = args.chain_of_thoughts
        ## loading input questions (and answer for reference accuracy computing)
        self.load_dataset(args)
        self.load_similarity()
        self.apikey_list = apikey_list
        self.apikey_idx = 0
        openai.api_key = self.apikey_list[self.apikey_idx]
        if args.engine == "opt":
            if args.with_six_gpus:
                self.initialize_opt_6gpu()
            elif args.with_one_gpu:
                self.initialize_opt_small()
            else:
                self.initialize_opt()
        elif args.engine == "llama":
            self.initialize_llama()
        elif args.engine == "bloom":
            from plm.bloom import get_model_and_tokenizer
            self.model, self.tokenizer = get_model_and_tokenizer(name="microsoft/bloom-deepspeed-inference-int8",
                                                                 dtype="int8")
        elif args.engine == "chat":
            import tiktoken
            self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        # elif args.engine == "codex":
        #     import tiktoken
        #     self.tokenizer = tiktoken.encoding_for_model(self.args.engine_name)
        # elif args.engine == "instruct":
        #     import tiktoken
        #     self.tokenizer = tiktoken.encoding_for_model("davinci-instruct-beta")
        elif args.engine == "chat-test":
            self.initialize_opt_small()
        elif args.engine in ['ada', 'babbage', 'curie', 'davinci', 'gpt3']:
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        else:
            import tiktoken
            self.tokenizer = tiktoken.encoding_for_model(self.args.engine_name)
            # self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        if self.args.use_blip2:
            if not self.args.with_blip2_api:
                blip2_model_name = "pretrain_flant5xl" if self.args.use_v100 else "pretrain_flant5xxl"
                from lavis.models import load_model_and_preprocess
                if args.engine == "chat-test":
                    self.blip2_device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
                    self.blip2_model, self.blip2_vis_processors, _ = load_model_and_preprocess(name="blip2_t5",
                                                                                            model_type=blip2_model_name,
                                                                                            is_eval=True,
                                                                                            device=self.blip2_device)
                    import pdb
                    pdb.set_trace()
                else:
                    self.blip2_device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
                    self.blip2_model, self.blip2_vis_processors, _ = load_model_and_preprocess(name="blip2_t5",
                                                                                            model_type=blip2_model_name,
                                                                                            is_eval=True,
                                                                                            device=self.blip2_device)
                print("Finish loading BLIP2 model")
            else:
                self.blip2_api = API_URLS = [ "http://localhost:5000/api/generate", ]

        if args.with_clip_verify or args.choice_only:
            model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16")
            model = model.cuda()
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
            self.clip_model, self.clip_processor = model, processor

        if args.oracle_attend:
            with open(self.args.val_sim_file, "rb") as fh:
                self.val_oracle_attend = pickle.load(fh)

        self.temp_question = "What is the person doing?"

    def find_image(self, img_key):
        img_full_path = os.path.join(self.raw_image_dir, "%012d.jpg" % img_key)
        #print(img_full_path)
        return Image.open(img_full_path).convert("RGB")

    def load_dataset(self, args):
        test_name = "test" if args.test_only else "val"
        self.raw_image_dir = os.path.join(self.args.raw_image_dir, "%s2017" % test_name)
        _, self.answer_dict, self.question_dict, self.rationale_dict, self.choices_dict = \
            self.load_anno(None, f'{args.coco_path}/aokvqa_v1p0_{test_name}.json',
                           f'{args.coco_path}/aokvqa_v1p0_{test_name}.json', choice_only=args.choice_only)
        self.val_keys = list(self.question_dict.keys())
        self.val_keys = self.val_keys[int(args.start * len(self.val_keys)):int(args.end * len(self.val_keys))] # OK

        ## load cached image representation (Coco caption & Tags)
        self.inputtext_dict = self.load_cachetext()

        self.traincontext_caption_dict, self.traincontext_answer_dict, \
        self.traincontext_question_dict, self.traincontext_rationale_dict, \
        self.traincontext_choices_dict = \
            self.load_anno('%s/captions_train2017.json' % args.coco_path, \
                           '%s/aokvqa_v1p0_train.json' % args.coco_path, \
                           '%s/aokvqa_v1p0_train.json' % args.coco_path, choice_only=args.choice_only)

        self.traincontext_interactive_answer_dict = self.traincontext_answer_dict
        self.traincontext_interactive_question_dict = self.traincontext_question_dict
        self.train_keys = list(self.traincontext_answer_dict.keys()) # OK
        self.train_interactive_keys = self.train_keys # OK

        if args.caption_type == 'vinvl_ocr':
            self.load_ocr(os.path.join(self.args.sg_path, "coco17_ocr_train.json"),
                          os.path.join(self.args.sg_path, f"coco17_ocr_{test_name}.json"),
                          os.path.join(self.args.sg_path, "scene_graph_coco17_attr"))
        self.sg_dir = os.path.join(self.args.sg_path, "scene_graph_coco17")
        self.sg_attr_dir = os.path.join(self.args.sg_path, "scene_graph_coco17_attr")
        self.sg_cap_dir = os.path.join(self.args.sg_path, self.args.concept_caption_path)

    def load_anno(self, coco_caption_file, answer_anno_file, question_anno_file, choice_only=False):
        if coco_caption_file is not None:
            coco_caption = json.load(open(coco_caption_file, 'r'))
            if type(coco_caption) == type({}): coco_caption = coco_caption['annotations']
        answer_anno = json.load(open(answer_anno_file, 'r'))
        question_anno = json.load(open(question_anno_file, 'r'))

        caption_dict = {}
        if coco_caption_file is not None:
            for sample in coco_caption:
                if sample['image_id'] not in caption_dict:
                    caption_dict[sample['image_id']] = [sample['caption']]
                else:
                    caption_dict[sample['image_id']].append(sample['caption'])

        answer_dict = {}
        for sample in answer_anno:
            if str(sample['image_id']) + '<->' + str(sample['question_id']) not in answer_dict:
                if choice_only:
                    if 'correct_choice_idx' in sample:
                        answer_dict[str(sample['image_id']) + '<->' + str(sample['question_id'])] = sample[
                            "correct_choice_idx"]
                    else:
                        answer_dict[str(sample['image_id']) + '<->' + str(sample['question_id'])] = 0
                else:
                    if 'direct_answers' in sample:
                        answer_dict[str(sample['image_id']) + '<->' + str(sample['question_id'])] = sample[
                            "direct_answers"]
                    else:
                        answer_dict[str(sample['image_id']) + '<->' + str(sample['question_id'])] = [""]

        question_dict = {}
        for sample in question_anno:
            if str(sample['image_id']) + '<->' + str(sample['question_id']) not in question_dict:
                question_dict[str(sample['image_id']) + '<->' + str(sample['question_id'])] = sample['question']

        rationales_dict = {}
        for sample in answer_anno:
            if str(sample['image_id']) + '<->' + str(sample['question_id']) not in rationales_dict:
                if 'rationales' in sample:
                    rationales_dict[str(sample['image_id']) + '<->' + str(sample['question_id'])] = sample['rationales']
                else:
                    rationales_dict[str(sample['image_id']) + '<->' + str(sample['question_id'])] = ""

        choices_dict = {}
        for sample in answer_anno:
            choices_dict[str(sample['image_id']) + '<->' + str(sample['question_id'])] = sample['choices']

        return caption_dict, answer_dict, question_dict, rationales_dict, choices_dict

    def sleep(self, sleep_time=1.5, switch_key=False):
        if self.args.engine == "codex":
            sleep_time = 0.1
        if switch_key:
            self.apikey_idx += 1
            if self.apikey_idx >= len(self.apikey_list):
                self.apikey_idx = 0
        openai.api_key = self.apikey_list[self.apikey_idx]
        time.sleep(sleep_time)

    def load_ocr(self, train_ocr, val_ocr, sg_path, thres=0.2):
        train_ocr_dict = json.load(open(train_ocr))
        val_ocr_dict = json.load(open(val_ocr))
        self.train_ocr_text = {}
        self.val_ocr_text = {}
        for key in train_ocr_dict:
            tmp_ocr_list = train_ocr_dict[key]
            ocr_text = {}
            if len(tmp_ocr_list) > 0:
                obj_list = json.load(open(os.path.join(sg_path, f"{key.split('_')[-1]}.json")))
                for tmp_ocr in tmp_ocr_list:
                    box = tmp_ocr["box"]
                    text = tmp_ocr["text"]
                    conf = tmp_ocr["conf"]
                    if conf > thres:
                        max_match_val = -1
                        max_match_obj = ""
                        box = [box[0][0], box[0][1], box[1][0], box[2][1]]
                        for obj in obj_list[0]:
                            if bounding_box_matching(box, obj['rect']) > max_match_val:
                                max_match_obj = obj['class']
                                max_match_val = bounding_box_matching(box, obj['rect'])
                        ocr_text[max_match_obj] = f"Text {text} is on the {max_match_obj}."
            self.train_ocr_text[int(key.split("_")[-1])] = ocr_text

        for key in val_ocr_dict:
            tmp_ocr_list = val_ocr_dict[key]
            ocr_text = {}
            if len(tmp_ocr_list) > 0:
                obj_list = json.load(open(os.path.join(sg_path, f"{key.split('_')[-1]}.json")))
                for tmp_ocr in tmp_ocr_list:
                    box = tmp_ocr["box"]
                    text = tmp_ocr["text"]
                    conf = tmp_ocr["conf"]
                    if conf > thres:
                        max_match_val = -1
                        max_match_obj = ""
                        box = [box[0][0], box[0][1], box[1][0], box[2][1]]
                        for obj in obj_list[0]:
                            if bounding_box_matching(box, obj['rect']) > max_match_val:
                                max_match_obj = obj['class']
                                max_match_val = bounding_box_matching(box, obj['rect'])
                        ocr_text[max_match_obj] = f"Text {text} is on the {max_match_obj}."
            self.val_ocr_text[int(key.split("_")[-1])] = ocr_text

    def initialize_llama(self):
        from transformers import LlamaForCausalLM, LlamaTokenizer
        self.model = LlamaForCausalLM.from_pretrained(self.args.llama_path,
                                                                    device_map="auto", torch_dtype=torch.float16)
        self.tokenizer = LlamaTokenizer.from_pretrained(self.args.llama_path)

    def initialize_opt_small(self):
        from transformers import OPTForCausalLM
        self.model = OPTForCausalLM.from_pretrained("facebook/opt-1.3b",
                                                    device_map="auto", torch_dtype=torch.float16)
        self.tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-1.3b")

    def initialize_opt(self):
        num_device = 8
        import math
        num_layers = math.ceil(64 / 8.0)
        assert torch.cuda.device_count() >= num_device
        opt_device_map = {'model.decoder.embed_tokens': 0,
                          'lm_head': 0,
                          'model.decoder.embed_positions': 0,
                          'model.decoder.final_layer_norm': 0,
                          'model.decoder.layers.0': 0}
        for layer in range(64):
            layer_name = "model.decoder.layers.%s" % (str(layer))
            device = layer // num_layers
            opt_device_map[layer_name] = device
        from transformers import OPTForCausalLM
        self.model = OPTForCausalLM.from_pretrained("facebook/opt-66b",
                                                    device_map=opt_device_map, torch_dtype=torch.float16)
        self.tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-66b")

    def initialize_opt_6gpu(self):
        num_device = 6
        import math
        num_layers = math.ceil(64 / 6.0)
        assert torch.cuda.device_count() >= num_device
        opt_device_map = {'model.decoder.embed_tokens': 0,
                          'lm_head': 0,
                          'model.decoder.embed_positions': 0,
                          'model.decoder.final_layer_norm': 0,
                          'model.decoder.layers.0': 0}
        for layer in range(64):
            layer_name = "model.decoder.layers.%s" % (str(layer))
            device = layer // num_layers
            opt_device_map[layer_name] = device
        from transformers import OPTForCausalLM
        self.model = OPTForCausalLM.from_pretrained("facebook/opt-66b",
                                                    device_map=opt_device_map, torch_dtype=torch.float16)
        self.tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-66b")

    def decode_scene_graph(self, sg_attr):
        attr_list = []
        for attr in sg_attr:
            if self.args.iterative_strategy == "sg":
                attr_list.append(f"{attr[1]} is {', '.join(attr[2])}")
            elif self.args.iterative_strategy == "caption":
                attr_list.append(attr[3])
                if self.args.caption_type == "vinvl_ocr" and attr[4] != "":
                    attr_list.append(attr[4])
            else:
                assert False

        text = ""
        text += "\n".join(attr_list)
        return text

    def inference(self, save_every_step):
        answers = []
        full_answers = []
        # i=0
        if save_every_step:
            os.system("mkdir -p %s" % self.args.output_path)
            os.system("mkdir -p %s/prompt_samples" % self.args.output_path)
            os.system("mkdir -p %s/format_samples" % self.args.output_path)

        if self.args.pick_example_with_question_mode:
            while True:
                image_id = input("Input one image id please")
                question = input("Input one question please")
                image_id = str(image_id)
                self.given_question = question
                for idx, key in enumerate(tqdm(self.val_keys)):
                    if image_id not in key:
                        continue
                    final_answer, answer_list = self.sample_inference_interactive(key)
                    print(final_answer)
                    print(answer_list)
                    pdb.set_trace()

        for idx, key in enumerate(tqdm(self.val_keys)):
            if save_every_step:
                out_file_name = "%s/prompt_samples/sample_%s_*.json" % (self.args.output_path, str(idx))
                print(out_file_name)
                out_file_list = glob.glob(out_file_name)
                if len(out_file_list) > 0:
                    continue
            if self.args.pick_example_mode:
                if not self.pick_example(key):
                    continue
            final_answer, answer_list = self.sample_inference_interactive(key)
            answers.append(final_answer)
            full_answers.append(answer_list)
            acc = 0.
            for answer in answers:
                acc += float(answer[3])
            print(acc * 100. / len(answers), len(answers))
            if save_every_step:
                json.dump(answers[-1], open("%s/prompt_samples/sample_%s_%s.json" % \
                                            (self.args.output_path, str(idx), str(float(answers[-1][3]))), 'w'))
                json.dump(full_answers[-1], open("%s/format_samples/sample_%s_%s.json" % \
                                            (self.args.output_path, str(idx), str(float(answers[-1][3]))), 'w'))
            # i += 1
            # if i > 3:
            #     break
        return answers, full_answers

    def sample_inference_interactive(self, key):
        img_key = int(key.split('<->')[0]) if self.args.set_name!="fvqa" else self.image_dict[key] # for fvqa
        raw_image = self.find_image(img_key)
        if self.args.use_blip2:
            if not self.args.with_blip2_api:
                self.current_blip2_image = self.blip2_vis_processors["eval"](raw_image).unsqueeze(0).to(
                    self.blip2_device)
            else:
                self.current_blip2_image = raw_image
        if self.args.debug:
            t1=time.time()
        if self.args.set_name!="fvqa":
            scene_graph_path = os.path.join(self.sg_attr_dir, str(img_key).zfill(12) + ".json")
        else:
            scene_graph_path = os.path.join(self.sg_attr_dir, str(img_key).replace(".jpg", "") + ".json")
        scene_graph_attr = json.load(open( scene_graph_path))
        if self.args.iterative_strategy == "caption":
            scene_path = os.path.join(self.sg_cap_dir, str(img_key).zfill(12) + ".json")
            if not os.path.isfile(scene_path):
                # backup caption
                scene_path = os.path.join(self.sg_cap_dir + "_v2", str(img_key).zfill(12) + ".json")
            if not os.path.isfile(scene_path):
                scene_graph_caption = [f"{attr['class']} is {', '.join(attr['attr'])}." \
                                           for attr in scene_graph_attr[0]]
            else:
                scene_graph_caption = json.load(open(scene_path))
        attr_list = []
        for attr_id, attr in enumerate(scene_graph_attr[0]):
            if self.args.iterative_strategy == "caption":
                if isinstance(scene_graph_caption, list):
                    tmp_cap = scene_graph_caption[attr_id]
                else:
                    rect_str = str(attr['rect'])
                    try:
                        tmp_cap = scene_graph_caption[rect_str]
                    except:
                        tmp_cap = attr['class']
                        print("Fail to parse attr\n")
                tmp_attr = [attr['conf'], attr['class'], attr['attr'], tmp_cap]
            else:
                tmp_attr = [attr['conf'], attr['class'], attr['attr']]
            if self.args.caption_type == "vinvl_ocr":
                if attr['class'] in self.val_ocr_text[img_key]:
                    tmp_attr.append(self.val_ocr_text[img_key][attr['class']])
                else:
                    tmp_attr.append("")
            attr_list.append(tmp_attr)
        attr_list.sort(key=lambda x: x[0], reverse=True)

        if self.args.use_blip2:
            attr_list_blip2 = []
            attr_name_list = []
            for attr in attr_list:
                if attr[1] not in attr_name_list:
                    attr_name_list.append(attr[1])
                    attr_list_blip2.append(attr)
            attr_list = attr_list_blip2
            self.current_global_caption = self.query_blip2_global_caption(self.question_dict[key])

        answer_list = []
        noticed_attr_list = []
        thoughts = []
        answer_text = ""
        if self.args.debug:
            t2=time.time()
            print("    PREPARE TIME", t2-t1)
        self.current_conversation = []
        rounds = 1 if self.args.all_regional_captions else self.args.rounds
        for i in range(rounds):
            if self.args.debug:
                t3=time.time()
            if self.args.random_attend:
                idx = random.randint(0, len(attr_list)-1)
            elif self.args.oracle_attend:
                attr_scores = [self.val_oracle_attend[key][attr[1]] for attr in attr_list]
                idx = attr_scores.index(max(attr_scores))
            elif self.args.all_regional_captions:
                idx = None
            else:
                idx = self.interactive(key, attr_list)
            # HERE
            torch.cuda.synchronize()
            if self.args.debug:
                t4=time.time()

            if self.args.use_blip2:
                if idx >= len(attr_list):
                    idx = idx % len(attr_list)
                    print("index %d, out of object list."%(idx))
                    print(attr_list)
                local_caption = self.query_blip2_local_caption(attr_list[idx][1], self.question_dict[key])
                noticed_attr_list.append([attr_list[idx][0], attr_list[idx][1], [], local_caption, ""])
            elif self.args.all_regional_captions:
                noticed_attr_list.extend([attr_i for attr_i in attr_list])
            else:
                noticed_attr_list.append(attr_list[idx])
            if self.chain_of_thoughts:
                answer_list.append(self.sample_inference(key, [] if self.args.ablation_visual else noticed_attr_list
                                                         , [] if self.args.ablation_reason else thoughts))
            else:
                answer_list.append(self.sample_inference(key, noticed_attr_list))
            if self.args.debug:
                t5=time.time()
                print("    VISUAL LOOP TIME", t4-t3)
                print("    REASON LOOP TIME", t5-t4)
            torch.cuda.synchronize()
            if idx != None:
                attr_list = attr_list[:idx] + attr_list[idx + 1:]
            thoughts.append(answer_list[-1][4])
            if answer_text == answer_list[-1][1]:
                break
            else:
                answer_text = answer_list[-1][1]
        final_answer = answer_list[-1]
        return final_answer, answer_list

    def pick_example(self, key):
        img_key = int(key.split('<->')[0]) if self.args.set_name != "fvqa" else self.image_dict[key]  # for fvqa
        scene_graph_path = os.path.join(self.sg_attr_dir, str(img_key).zfill(12) + ".json")
        scene_graph_attr = json.load(open(scene_graph_path))
        for attr_id, attr in enumerate(scene_graph_attr[0]):
            if attr['class'] in ['girl', 'boy', 'man', 'woman'] and len(attr['attr']) > 0:
                description = attr['attr'][0]
                self.temp_question = f"What is the {description} {attr['class']} doing?"
                return True
        return False

    def query_blip2_basic(self, image, prompt, use_pred_answer=False):
        if not self.args.with_blip2_api:
            if use_pred_answer:
                output = self.blip2_model.predict_answers({"image": image, "text_input": prompt}, max_len=25)
            else:
                output = self.blip2_model.generate({"image": image, "text_input": prompt})
            if self.args.debug:
                import pdb
                pdb.set_trace()
        else:
            # api only support predict_answers CALL
            output = utils_api.blip_completev2(images=[image ], texts= [prompt], blip_urls=self.blip2_api, num_beams=5, length_penalty=-1.0, encoding_format="PNG",)
            return output

        return output

    def query_blip2_objects(self):
        obj_list = []
        max_obj_num = 10
        while len(obj_list) < max_obj_num:
            if len(obj_list)==0:
                tmp_obj_name_list = self.query_blip2_basic(image=self.current_blip2_image,
                                         prompt="Give me the name of one object, creature, or entity in the image.")
            else:
                tmp_prompt = "Give me the name of one object, creature, or entity in the image besides"
                for tmp_idx, tmp_name in enumerate(obj_list):
                    tmp_prompt = tmp_prompt +" %s"%tmp_name
                    if tmp_idx < len(obj_list) -1:
                        tmp_prompt +=","
                    else:
                        tmp_prompt +="?"
                    tmp_obj_name_list = self.query_blip2_basic(image=self.current_blip2_image, prompt=tmp_prompt)

            tmp_obj_name_list_refine = parge_obj_name(tmp_obj_name_list)
            print(tmp_obj_name_list_refine)

            all_exist_flag = True
            for obj_name in tmp_obj_name_list_refine:
                if obj_name not in obj_list:
                    obj_list.append(obj_name)
                    all_exist_flag = False
            if all_exist_flag:
                break
        obj_list = list(set(obj_list))
        attr_list = [[1.0, obj_name] for obj_name in obj_list] # [[confidence, name]]
        print(attr_list)
        if self.args.debug:
            pdb.set_trace()
        return attr_list

    def query_blip2_global_caption(self, question):
        global_caption = self.query_blip2_basic(image=self.current_blip2_image, prompt="An image of ")[0]
        global_caption_question = self.query_blip2_basic(image=self.current_blip2_image, prompt=f"Question: Please "
                                                   f"look at the picture and answer the following question. "
                                                   f"{question} Answer:", use_pred_answer=True)[0]
        if self.args.debug:
            print(". ".join([global_caption, global_caption_question]))
        return ". ".join([global_caption, global_caption_question])

    def query_blip2_local_caption(self, obj_name, question):
        local_caption_raw = self.query_blip2_basic(image=self.current_blip2_image,
                                                   prompt=f"Question: Look at the {obj_name} in this image. Please give a detailed "
                                                          f"description of the {obj_name} in this image. Answer:", use_pred_answer=True)[0]
        if self.args.engine == "chat":
            self.current_conversation.append({
                'role': 'user',
                'content': f'You will to look at the {obj_name} in the picture and find {local_caption_raw}.'
                           f'To find the answer to {question}, you can ask one question about the {obj_name}. '
                           f'Please tell me the question you want to ask directly.'
            })
            successful = False
            while not successful:
                try:
                    self.sleep()
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=self.current_conversation,
                        max_tokens=40,
                        temperature=0.,
                        stream=False,
                    )
                    successful = True
                except Exception as e:
                    print(e)
                    self.sleep(switch_key=True)
            question_from_chatgpt = response['choices'][0]['message']['content']
        elif self.args.engine == "chat-test":
            self.current_conversation.append({
                'role': 'user',
                'content': f'You will to look at the {obj_name} in the picture and find {local_caption_raw}.'
                           f'To find the answer to {question}, you can ask one question about the {obj_name}. '
                           f'Please tell me the question you want to ask directly.'
            })
            question_from_chatgpt = "Who are you?"
        elif self.args.engine in ["ada", "babbage", "curie", "davinci", "codex", "instruct"]:
            prompt = f"I look at the {obj_name} in the picture and find {local_caption_raw}. " \
                     f"To find the answer to {question}, I ask one question about the {obj_name}. " \
                     f"My question is:"
            successful = False
            while not successful:
                try:
                    self.sleep()
                    response = openai.Completion.create(
                        engine=self.args.engine_name,
                        prompt=prompt,
                        max_tokens=41,
                        logprobs=1,
                        temperature=0.,
                        stream=False,
                        stop=["<|endoftext|>", "?", " ?"],
                    )
                    successful = True
                except Exception as e:
                    print(e)
                    self.sleep(switch_key=True)
            question_from_chatgpt = response['choices'][0]['text'].strip()+ "?"
        else:
            question_from_chatgpt = "Empty"
        local_caption_question = self.query_blip2_basic(image=self.current_blip2_image,
                                                        prompt=f"Question: Please look at the {obj_name} and answer the following question."
                                                                             f" {question_from_chatgpt} Answer:", use_pred_answer=True)[0]
        local_caption = ". ".join([local_caption_raw, question_from_chatgpt + " The answer is " + local_caption_question])
        if self.args.debug:
            print(local_caption)
            pdb.set_trace()
        return local_caption

    def query_blip2_thought_match_img(self, thought):
        blip2_answer = self.query_blip2_basic(image=self.current_blip2_image,
                                              prompt=f"Question: Does this sentence match the facts in the picture? Please answer yes or no. "
                                                     f"Sentence: In this picture, {thought} Answer:")[0]
        if self.args.debug:
            print(blip2_answer, thought)
        if blip2_answer == "no":
            correction = self.query_blip2_basic(image=self.current_blip2_image,
                                                prompt=f"Question: Please correct the following sentence according to "
                                                       f"the image. Sentence: {thought}")[0]
            return correction
        else:
            return thought

    def interactive(self, key, attr_list):
        context_key_list, rel_obj_list = self.get_interactive_context_keys(key, \
                                                                           self.args.similarity_metric,
                                                                           self.args.n_shot)
        question = self.question_dict[key]
        if self.args.pick_example_mode:
            question = self.temp_question
        if self.args.pick_example_with_question_mode:
            question = self.given_question
        if self.args.engine == "chat" or self.args.engine == "chat-test":
            system_prompt = "Let's play a game. I have an image and a complex question about it, and you will give me the " \
                     "name of object in the image you want to look at the most. Please follow the format" \
                     " of the following examples and give me an object name directly.\n"
            prompt = "===\n"
        else:
            prompt = 'Please select the object most related to the question.\n===\n'
        for ni in range(self.args.n_shot):
            context_key = context_key_list[ni]
            rel_obj = rel_obj_list[ni]
            rel_obj = [k for k, v in sorted(rel_obj.items(), key=lambda item: item[1], reverse=True)]
            if self.args.set_name=="fvqa":
                img_context_key = self.image_dict[context_key]
            else:
                img_context_key = int(context_key.split('<->')[0])
            while True:  ## make sure get context with valid question and answer
                if self.args.choice_only or (len(self.traincontext_interactive_question_dict[context_key]) != 0 and len(
                        self.traincontext_interactive_answer_dict[context_key][0]) != 0):
                    break
                context_key = self.train_interactive_keys[random.randint(0, len(self.train_interactive_keys) - 1)]
            if self.args.set_name=="fvqa":
                context_scene_graph = json.load(open(os.path.join(self.sg_dir, str(img_context_key).replace(".jpg", "") + ".json")))
                context_scene_graph_attr = json.load(open(os.path.join(self.sg_attr_dir, str(img_context_key).replace(".jpg", "") + ".json")))
            else:
                context_scene_graph = json.load(open(os.path.join(self.sg_dir, str(img_context_key).zfill(12) + ".json")))
                context_scene_graph_attr = json.load(open(os.path.join(self.sg_attr_dir, str(img_context_key).zfill(12) + ".json")))

            obj_list = []
            conf_list = []
            for obj in context_scene_graph[0]:
                if obj['class'] not in obj_list:
                    obj_list.append(obj['class'])
                    conf_list.append(obj['conf'])
            for obj in context_scene_graph_attr[0]:
                if obj['class'] not in obj_list:
                    obj_list.append(obj['class'])
                    conf_list.append(obj['conf'])
            if self.args.engine == "chat" or self.args.engine == "chat-test":
                prompt += 'Question: %s\n===\n' % (self.traincontext_interactive_question_dict[context_key])
            elif self.args.use_attributes_to_see or self.args.use_caption_to_see:
                prompt += 'Question: %s\n===\n' % (self.traincontext_interactive_question_dict[context_key])
            else:
                prompt += 'Question: %s\n===\nOptions:\n' % (self.traincontext_interactive_question_dict[context_key])
            prompt += "The most related option is %s.\n\n===\n" % rel_obj[0]
        obj_list = [obj[1] for obj in attr_list]
        if self.args.engine == "chat" or self.args.engine == "chat-test":
            prompt += "Question: %s\n===\nOptions: %s\n" % (question, ", ".join(obj_list))
        elif self.args.use_attributes_to_see:
            obj_list = [f"{obj[1]}: {' '.join(obj[2])} {obj[1]}" for obj in attr_list]
            prompt += "Question: %s\n===\nOptions: %s\n" % (question, ",\n".join(obj_list))
        elif self.args.use_caption_to_see:
            obj_list = [f"{obj[1]}: {obj[-2]}" for obj in attr_list]
            prompt += "Question: %s\n===\nOptions: %s\n" % (question, ", ".join(obj_list))
        else:
            prompt += "Question: %s\n===\nOptions:\n" % (question)
        prompt += "The most related option is"
        if self.args.engine in ["ada", "babbage", "curie", "davinci", "codex", "instruct", "gpt3"]:
            obj_idx_list = []
            for obj in obj_list:
                obj_idx_list.append(self.tokenizer.encode(f" {obj}")[0])
            logit_bias = {}
            current_bias = 100
            successful = False
            if self.args.engine == "codex":
                engine_name = "code-davinci-002"
            elif self.args.engine == "instruct":
                engine_name = "davinci-instruct-beta"
            elif self.args.engine == "gpt3":
                engine_name = "text-davinci-001"
            else:
                engine_name = self.args.engine
            while not successful:
                for tok_idx in obj_idx_list:
                    logit_bias[str(tok_idx)] = current_bias
                try:
                    self.sleep()
                    response = openai.Completion.create(
                        engine=engine_name,
                        prompt=prompt,
                        max_tokens=4,
                        logprobs=1,
                        temperature=0.,
                        stream=False,
                        stop=["\n", "<|endoftext|>"],
                        logit_bias=logit_bias
                    )
                    successful = True
                except Exception as e:
                    print(e)
                    self.sleep(switch_key=True)
            result = self.tokenizer.encode(response['choices'][0]['text'])[0]
            if result in obj_idx_list:
                result = obj_idx_list.index(result)
            else:
                result = 0
        elif self.args.engine == "chat":
            obj_idx_list = []
            for obj in obj_list:
                obj_idx_list.append(self.tokenizer.encode(f" {obj}")[0])
            logit_bias = {}
            current_bias = 25
            successful = False
            while not successful:
                for tok_idx in obj_idx_list:
                    logit_bias[str(tok_idx)] = current_bias
                try:
                    self.sleep()
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}],
                        max_tokens=5,
                        temperature=0.,
                        stream=False,
                        # stop=["\n", "<|endoftext|>"],
                        logit_bias=logit_bias
                    )
                    successful = True
                except Exception as e:
                    print(e)
                    print(prompt)
                    current_bias = int(0.8 * current_bias)
                    self.sleep(switch_key=True)
            result = self.tokenizer.encode(response['choices'][0]['message']['content'])[0]
            if result in obj_idx_list:
                result = obj_idx_list.index(result)
            else:
                result = 0
            self.current_conversation = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response['choices'][0]['message']['content']}
            ]
        elif self.args.engine == "chat-test":
            print([{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}])
            self.current_conversation = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "Object0"}
            ]
            pdb.set_trace()
            result = 0
        elif self.args.engine == "opt" or self.args.engine == "llama":
            obj_idx_list = []
            for obj in obj_list:
                obj_idx_list.append(self.tokenizer.encode(f" {obj}")[1])
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(inputs.input_ids.to(torch.cuda.current_device()), max_length=len(inputs.input_ids[0]) + 5,
                                          return_dict_in_generate=True, output_scores=True)
            scores = outputs['scores']
            scores = scores[0][0][obj_idx_list]
            if self.args.use_attributes_to_see or self.args.use_caption_to_see:
                result_str = self.tokenizer.decode(outputs['sequences'][0][len(inputs.input_ids[0]):]).split("\n")[0].strip()
                result_str = result_str[:-1] if result_str.endswith(".") else result_str
                result = -1
                for obj_id, obj in enumerate(obj_list):
                    if result_str in obj:
                        result = obj_id
                if result == -1:
                    result = scores.argmax().item()
            else:
                result = scores.argmax().item()
        elif self.args.engine == "bloom":
            obj_idx_list = []
            for obj in obj_list:
                obj_idx_list.append(self.tokenizer.encode(f" {obj}")[0])
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(inputs.input_ids.to(torch.cuda.current_device()),
                                          max_new_tokens=5, return_dict_in_generate=True, output_scores=True)
            scores = outputs['scores']
            scores = scores[0][0][obj_idx_list]
            result = scores.argmax().item()
        else:
            assert False
        if self.args.debug:
            pdb.set_trace()
        return result

    def make_choices_text(self, choices, answer):
        return f"{', '.join(choices)}.", choices[answer]

    def sample_inference(self, key, scene_graph_attr, thoughts_list=None):
        img_key = int(key.split('<->')[0]) if self.args.set_name!="fvqa" else self.image_dict[key] # for fvqa
        if self.args.random_caption:
            random.seed(img_key) # keep random context in every step of the same sample consistent
        question, answer, caption = self.question_dict[key], self.answer_dict[key], self.inputtext_dict[img_key]
        if self.args.pick_example_mode:
            question = self.temp_question
        if self.args.pick_example_with_question_mode:
            question = self.given_question
        if self.args.random_caption:
            caption = random.choice(list(self.traincontext_caption_dict.values()))
        if self.args.use_blip2:
            caption_i = self.current_global_caption
        else:
            caption_i = caption[random.randint(0,
                               len(caption) - 1)]  ## select one caption if exists multiple, not true except COCO GT (5)

        default_sg_text = self.decode_scene_graph(scene_graph_attr)

        pred_answer_list, pred_prob_list, thought_list, all_thought_list = [], [], [], []
        context_key_list = self.get_context_keys(key, self.args.similarity_metric, self.args.n_shot * self.args.n_ensemble)



        for repeat in range(self.args.n_ensemble):
            if self.args.debug:
                t1=time.time()
            if self.args.engine == "chat" or self.args.engine == "chat-test":
                prompt_before_answer = "Based on the given information, I must guess the most possible answer. Answer:\n"
                system_prompt = "Let's play a game. I have an image and a complex question about it. I will provide you some information about" \
                                " the image in the context, and you will give me the possible answer and reason to the question. You must provide an answer and can not say unclear or unknown. " \
                                "Please follow the format and answer style of the following examples and complete the last example.\n"
                prompt = "===\n"
            else:
                prompt_before_answer = "Answer: The answer is"
                prompt = 'Please answer the question according to the above context.\n===\n'
            ## prompt format following GPT-3 QA API
            cur_caption_i = "" if self.args.remove_caption else caption_i
            for ni in range(self.args.n_shot):
                if context_key_list is None:
                    context_key = self.train_keys[random.randint(0, len(self.train_keys) - 1)]
                else:
                    context_key = context_key_list[ni + self.args.n_shot * repeat]
                while True:  ## make sure get context with valid question and answer
                    if self.args.choice_only or (len(self.traincontext_question_dict[context_key]) != 0 and len(
                            self.traincontext_answer_dict[context_key][0]) != 0):
                        break
                    context_key = self.train_keys[random.randint(0, len(self.train_keys) - 1)]
                img_context_key = int(context_key.split('<->')[0]) if self.args.set_name!="fvqa" else self.image_dict[context_key] # for fvqa
                if self.args.random_caption:
                    context_caption = random.choice(list(self.traincontext_caption_dict.values()))
                    context_caption = random.choice(context_caption)
                elif self.args.remove_caption:
                    context_caption = ""
                else:
                    context_caption = self.traincontext_caption_dict[img_context_key][
                              random.randint(0, len(self.traincontext_caption_dict[img_context_key]) - 1)]
                prompt += 'Context: %s\n===\n' % (context_caption)
                if self.args.choice_only:
                    choice_text, answer_text = self.make_choices_text(self.traincontext_choices_dict[context_key],
                                                                      self.traincontext_answer_dict[context_key])
                    choice_text = f"\nChoices: {choice_text}"
                else:
                    choice_text = ""
                    answer_text = self.traincontext_answer_dict[context_key][0]
                    #if self.args.set_name !="fvqa" else self.traincontext_answer_dict[context_key]
                if self.chain_of_thoughts:
                    rationale_text = self.traincontext_rationale_dict[context_key][0]
                    #if self.args.set_name !="fvqa" else self.traincontext_rationale_dict[context_key]
                    prompt += 'Question: %s%s\n%s %s. %s\n\n===\n' % (self.traincontext_question_dict[context_key],
                                                                             choice_text, prompt_before_answer, answer_text, rationale_text)
                else:
                    prompt += 'Question: %s%s\n%s %s\n\n===\n' % (
                    self.traincontext_question_dict[context_key], choice_text, prompt_before_answer, answer_text)

            if thoughts_list is not None and len(thoughts_list) > 0:
                cur_thoughts_list = [th for th in thoughts_list if th != '']
                if len(cur_thoughts_list) > 0:
                    cur_caption_i += "\n"
                    cur_caption_i += " ".join(cur_thoughts_list)
            if self.args.choice_only:
                choice_text, _ = self.make_choices_text(self.choices_dict[key], 0)
                choice_text = f"\nChoices: {choice_text}"
            else:
                choice_text = ""
            if default_sg_text == "":
                prompt += 'Context: %s\n===\n' % cur_caption_i
            else:
                prompt += 'Context: %s\n%s\n===\n' % (cur_caption_i, default_sg_text)

            if self.chain_of_thoughts:
                prompt += 'Question: %s%s\n%s' % (question, choice_text, prompt_before_answer)
            else:
                prompt += 'Question: %s%s\n%s' % (question, choice_text, prompt_before_answer)
            response = None
            if self.args.debug:
                t2=time.time()
            if self.args.engine in ["ada", "babbage", "curie", "davinci", "codex", "instruct", "gpt3"]:
                successful = False
                if self.args.engine == "codex":
                    engine_name = "code-davinci-002"
                elif self.args.engine == "instruct":
                    engine_name = "davinci-instruct-beta"
                elif self.args.engine == "gpt3":
                    engine_name = "text-davinci-001"
                else:
                    engine_name = self.args.engine
                while not successful:
                    try:
                        self.sleep()
                        response = openai.Completion.create(
                            engine=engine_name,
                            prompt=prompt,
                            max_tokens=41,
                            logprobs=1,
                            temperature=0.,
                            stream=False,
                            stop=["\n", "<|endoftext|>"]
                        )
                        successful = True
                    except Exception as e:
                        print(e)
                        self.sleep(switch_key=True)
                plist = []
                if self.chain_of_thoughts:
                    for ii in range(len(response['choices'][0]['logprobs']['tokens'])):
                        if response['choices'][0]['logprobs']['tokens'][ii].endswith("."):
                            break
                        plist.append(response['choices'][0]['logprobs']['token_logprobs'][ii])
                    pred_answer_list.append(process_answer(response['choices'][0]["text"].split(".")[0]))
                    thought = ".".join(response['choices'][0]["text"].split(".")[1:]).strip()
                    pred_prob_list.append(sum(plist))
                else:
                    for ii in range(len(response['choices'][0]['logprobs']['tokens'])):
                        if response['choices'][0]['logprobs']['tokens'][ii] == '\n':
                            break
                        plist.append(response['choices'][0]['logprobs']['token_logprobs'][ii])
                    pred_answer_list.append(process_answer(response['choices'][0]["text"]))
                    pred_prob_list.append(sum(plist))
            elif self.args.engine == "chat":
                successful = False
                while not successful:
                    try:
                        self.sleep()
                        response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": prompt}
                            ],
                            max_tokens=40,
                            temperature=0.,
                            stream=False,
                        )
                        successful = True
                    except Exception as e:
                        print(e)
                        print(prompt)
                        self.sleep(switch_key=True)
                if self.chain_of_thoughts:
                    pred_answer_list.append(process_answer(response['choices'][0]['message']['content'].split(".")[0]))
                    thought = ".".join(response['choices'][0]['message']['content'].split(".")[1:]).strip()
                    pred_prob_list.append(0)
                else:
                    pred_answer_list.append(process_answer(response['choices'][0]['message']['content']))
                    pred_prob_list.append(0)
            elif self.args.engine == "chat-test":
                print([{"role": "system", "content": system_prompt},{"role": "user", "content": prompt}])
                pdb.set_trace()
                pred_answer_list.append("fake answer")
                thought = "This is a fake thought."
                pred_prob_list.append(0)
            elif self.args.engine == "opt" or self.args.engine == "llama" or self.args.engine == "bloom":
                inputs = self.tokenizer(prompt, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.model.generate(inputs.input_ids.to(torch.cuda.current_device()), max_length=len(inputs.input_ids[0]) + 40,
                                                  return_dict_in_generate=True, output_scores=True)
                    plist = []
                    result = self.tokenizer.batch_decode(outputs['sequences'][:, len(inputs.input_ids[0]):])[0]
                    if self.chain_of_thoughts:
                        for ii in range(len(inputs.input_ids[0]), len(outputs['sequences'][0])):
                            tok = outputs['sequences'][0][ii]
                            if self.tokenizer.decode([tok]) == '.':
                                break
                            scores = torch.log_softmax(outputs['scores'][ii - len(inputs.input_ids[0])], dim=-1)
                            plist.append(scores[0][tok])
                        thought = ".".join(result.split("\n")[0].split("The answer is")[-1].split(".")[1:]).strip()
                        pred_answer = process_answer(result.split("\n")[0].split("The answer is")[-1].split(".")[0])
                        pred_answer_list.append(pred_answer)
                        pred_prob_list.append(sum(plist))
                    else:
                        for ii in range(len(inputs.input_ids[0]), len(outputs['sequences'][0])):
                            tok = outputs['sequences'][0][ii]
                            if self.tokenizer.decode([tok]) == '\n':
                                break
                            scores = torch.log_softmax(outputs['scores'][ii - len(inputs.input_ids[0])], dim=-1)
                            plist.append(scores[0][tok])
                        pred_answer = process_answer(result.split("\n")[0])
                        pred_answer_list.append(pred_answer)
                        pred_prob_list.append(sum(plist))
            else:
                assert False
            if self.args.debug:
                t3=time.time()
            if self.chain_of_thoughts and self.args.with_clip_verify:
                if self.args.use_blip2:
                    tmp_thought_list = thought.split(".")
                    new_tmp_thought_list = []
                    new_tmp_thought_list_all = []
                    for thought in tmp_thought_list:
                        new_tmp_thought_list.append(self.query_blip2_thought_match_img(thought))
                        new_tmp_thought_list_all.append(thought)
                    new_thought = ".".join(new_tmp_thought_list).strip() + "."
                    new_thought_all = ".".join(new_tmp_thought_list_all).strip() + "."
                    if len(new_tmp_thought_list) > 0:
                        thought_list.append(new_thought)
                    else:
                        thought_list.append('')
                    all_thought_list.append(new_thought_all)
                else:
                    with torch.no_grad():
                        img_id = self.valkey2idx[key]
                        img_emb = torch.from_numpy(self.image_val_feature[img_id]).cuda().float().unsqueeze(dim=0)
                        tmp_thought_list = thought.split(".")
                        inputs = self.clip_processor(text=tmp_thought_list, return_tensors="pt", padding=True)
                        inputs = {k: v.cuda() for k, v in inputs.items()}
                        clip_outputs = self.clip_model(**inputs)
                        thought_emb = clip_outputs['pooler_output']
                        thought_emb /= thought_emb.norm(dim=-1, keepdim=True)
                        img_emb /= img_emb.norm(dim=-1, keepdim=True)
                        sim_cands = img_emb @ thought_emb.T
                        sim_thre = self.args.verify_threshold
                        new_tmp_thought_list = []
                        new_tmp_thought_list_all = []
                        for tid in range(sim_cands.shape[1]):
                            sim = sim_cands[0, tid].item()
                            if sim > sim_thre and len(tmp_thought_list[tid]) > 0:
                                new_tmp_thought_list.append(tmp_thought_list[tid])
                            new_tmp_thought_list_all.append(tmp_thought_list[tid])
                        new_thought = ".".join(new_tmp_thought_list).strip() + "."
                        new_thought_all = ".".join(new_tmp_thought_list_all).strip() + "."
                        if self.args.random_rationale:
                            new_thought = random.choice(list(self.traincontext_rationale_dict.values()))
                            new_thought = random.choice(new_thought)
                            new_tmp_thought_list = new_thought.split(".")
                            new_thought_all = random.choice(list(self.traincontext_rationale_dict.values()))
                            new_thought_all = random.choice(new_thought_all)
                        elif self.args.oracle_rationale:
                            new_thought = self.rationale_dict[key][0]
                            new_tmp_thought_list = new_thought.split(".")
                            new_thought_all = self.rationale_dict[key][0]
                        if len(new_tmp_thought_list) > 0:
                            thought_list.append(new_thought)
                        else:
                            thought_list.append('')
                        all_thought_list.append(new_thought_all)
            elif self.chain_of_thoughts:
                if self.args.random_rationale:
                    assert False
                thought_list.append(thought)
                all_thought_list.append(new_thought)
            if self.args.debug:
                t4=time.time()
                print("    REASON PREPARE TIME", t2-t1)
                print("    REASON INF TIME", t3-t2)
                print("    REASON POST TIME", t4-t3)
        maxval = -999.
        for ii in range(len(pred_prob_list)):
            if pred_prob_list[ii] > maxval:
                if self.chain_of_thoughts:
                    thoughts, all_thoughts = thought_list[ii], all_thought_list[ii]
                maxval, pred_answer = pred_prob_list[ii], pred_answer_list[ii]
        ## a rough accuracy estimator for fast results check
        if self.args.choice_only:
            if pred_answer not in self.choices_dict[key]:
                choices_list = self.choices_dict[key] + [pred_answer]
                inputs = self.clip_processor(text=choices_list, return_tensors="pt", padding=True)
                inputs = {k: v.cuda() for k, v in inputs.items()}
                clip_outputs = self.clip_model(**inputs)
                thought_emb = clip_outputs['pooler_output']
                thought_emb /= thought_emb.norm(dim=-1, keepdim=True)
                sim = thought_emb[-1].unsqueeze(0) @ thought_emb[:-1].T
                pred_answer = self.choices_dict[key][sim.argmax().item()]
            final_score = 1 if pred_answer == self.choices_dict[key][answer] else 0
        else:
            counter = 0
            for ii in range(len(answer)):
                if pred_answer == answer[ii]: counter += 1
            final_score = min(1., float(counter) * 0.3)
        if self.args.debug:
            print(prompt)
            print(pred_answer)
            print(answer)
            pdb.set_trace()
        if self.chain_of_thoughts:
            return [key, pred_answer, prompt, final_score, thoughts, all_thoughts, float(maxval),
                    [attr[1] for attr in scene_graph_attr]]
        return [key, pred_answer, prompt, final_score, float(maxval), [attr[1] for attr in scene_graph_attr]]

    def get_context_keys(self, key, metric, n):
        if metric == 'question':
            lineid = self.valkey2idx[key]
            if self.args.pick_example_mode:
                inputs = self.clip_processor(text=[self.temp_question], return_tensors="pt", padding=True)
                inputs = {k: v.cuda() for k, v in inputs.items()}
                clip_outputs = self.clip_model(**inputs)
                val_feature = clip_outputs['pooler_output'].cpu()
                val_feature /= val_feature.norm(dim=-1, keepdim=True)
                similarity = np.matmul(self.train_feature, val_feature.detach()[0].numpy())
            else:
                similarity = np.matmul(self.train_feature, self.val_feature[lineid, :])
            index = similarity.argsort()[-n:][::-1]
            return [self.train_idx[str(x)] for x in index]
        elif metric == 'imagequestion':
            ## combined with Q-similairty (image+question)
            lineid = self.valkey2idx[key]
            if self.args.pick_example_mode:
                inputs = self.clip_processor(text=[self.temp_question], return_tensors="pt", padding=True)
                inputs = {k: v.cuda() for k, v in inputs.items()}
                clip_outputs = self.clip_model(**inputs)
                val_feature = clip_outputs['pooler_output'].cpu()
                val_feature /= val_feature.norm(dim=-1, keepdim=True)
                question_similarity = np.matmul(self.train_feature, val_feature.detach()[0].numpy())
            else:
                question_similarity = np.matmul(self.train_feature, self.val_feature[lineid, :])
            ## end of Q-similairty
            similarity = question_similarity + np.matmul(self.image_train_feature, self.image_val_feature[lineid, :])
            index = similarity.argsort()[-n:][::-1]
            return [self.train_idx[str(x)] for x in index]
        else:
            return None

    def get_related_obj_dict(self, key):
        if self.args.train_sim_metric == "rationale":
            return self.get_related_obj_dict_rationale(key)
        elif self.args.train_sim_metric == "answer":
            if not hasattr(self, "train_object_select"):
                self.train_object_select = pickle.load(open(self.args.train_sim_file, "rb"))
            return self.train_object_select[key]

    def get_related_obj_dict_rationale(self, key):
        img_context_key = int(key.split('<->')[0])
        context_scene_graph = json.load(open(os.path.join(self.sg_dir, str(img_context_key).zfill(12) + ".json")))
        context_scene_graph_attr = json.load(
            open(os.path.join(self.sg_attr_dir, str(img_context_key).zfill(12) + ".json")))

        obj_list = []
        for obj in context_scene_graph[0]:
            if obj['class'] not in obj_list:
                obj_list.append(obj['class'])
        for obj in context_scene_graph_attr[0]:
            if obj['class'] not in obj_list:
                obj_list.append(obj['class'])

        related_obj_dict = {}
        rationale = self.traincontext_rationale_dict[key]
        for obj in obj_list:
            for r in rationale:
                if obj in r:
                    if obj not in related_obj_dict:
                        related_obj_dict[obj] = 1
                    else:
                        related_obj_dict[obj] += 1
        return related_obj_dict

    def get_interactive_context_keys(self, key, metric, n):
        if metric == 'question':
            assert False
        elif metric == 'imagequestion':
            ## combined with Q-similairty (image+question)
            lineid = self.valkey2idx[key]
            if self.args.pick_example_mode:
                inputs = self.clip_processor(text=[self.temp_question], return_tensors="pt", padding=True)
                inputs = {k: v.cuda() for k, v in inputs.items()}
                clip_outputs = self.clip_model(**inputs)
                val_feature = clip_outputs['pooler_output'].cpu()
                val_feature /= val_feature.norm(dim=-1, keepdim=True)
                question_similarity = np.matmul(self.train_feature, val_feature.detach()[0].numpy())
            else:
                question_similarity = np.matmul(self.train_feature, self.val_feature[lineid, :])
            ## end of Q-similairty
            similarity = question_similarity + np.matmul(self.image_train_feature, self.image_val_feature[lineid, :])
            similarity = similarity.argsort()
            idx_list = []
            rel_obj_list = []
            for i in range(len(similarity)):
                context_key = self.train_idx[str(similarity[-1 - i])]
                rel_obj_dict = self.get_related_obj_dict(context_key)
                if len(rel_obj_dict) > 0:
                    idx_list.append(context_key)
                    rel_obj_list.append(rel_obj_dict)
                if len(idx_list) >= n:
                    break
            return idx_list, rel_obj_list
        else:
            return None

    def load_similarity(self):
        split = "test" if self.args.test_only else "val"
        val_idx = json.load(open('%s/aokvqa_qa_line2sample_idx_%s2017.json' % (self.args.similarity_path, split), 'r'))
        self.valkey2idx = {}
        for ii in val_idx:
            self.valkey2idx[val_idx[ii]] = int(ii)
        if self.args.similarity_metric == 'question':
            self.train_feature = np.load(
                '%s/coco_clip_vitb16_train2017_aokvqa_question.npy' % self.args.similarity_path)
            self.val_feature = np.load('%s/coco_clip_vitb16_%s2017_aokvqa_question.npy' % (self.args.similarity_path, split))
            self.train_idx = json.load(
                open('%s/aokvqa_qa_line2sample_idx_train2017.json' % self.args.similarity_path, 'r'))
        elif self.args.similarity_metric == 'imagequestion':
            self.train_feature = np.load(
                '%s/coco_clip_vitb16_train2017_aokvqa_question.npy' % self.args.similarity_path)
            self.val_feature = np.load('%s/coco_clip_vitb16_%s2017_aokvqa_question.npy' % (self.args.similarity_path, split))
            self.train_idx = json.load(
                open('%s/aokvqa_qa_line2sample_idx_train2017.json' % self.args.similarity_path, 'r'))
            self.image_train_feature = np.load(
                '%s/coco_clip_vitb16_train2017_aokvqa_convertedidx_image.npy' % self.args.similarity_path)
            self.image_val_feature = np.load(
                '%s/coco_clip_vitb16_%s2017_aokvqa_convertedidx_image.npy' % (self.args.similarity_path, split))

    def load_tags(self):
        tags_dict = {}
        tagging_pred_file = '%s/test.score.json.tsv' % self.args.tag_path
        read_tsv = csv.reader(open(tagging_pred_file, 'r'), delimiter="\t")
        for row in read_tsv:
            image_id, tags = int(row[0]), json.loads(row[1])
            tag_str = ', '.join([x['class'] for x in tags])
            tags_dict[image_id] = tag_str
        tagging_pred_file = '%s/val.score.json.tsv' % self.args.tag_path
        read_tsv = csv.reader(open(tagging_pred_file, 'r'), delimiter="\t")
        for row in read_tsv:
            image_id, tags = int(row[0]), json.loads(row[1])
            tag_str = ', '.join([x['class'] for x in tags])
            tags_dict[image_id] = tag_str
        tagging_pred_file = '%s/train.score.json.tsv' % self.args.tag_path
        read_tsv = csv.reader(open(tagging_pred_file, 'r'), delimiter="\t")
        for row in read_tsv:
            image_id, tags = int(row[0]), json.loads(row[1])
            tag_str = ', '.join([x['class'] for x in tags])
            tags_dict[image_id] = tag_str
        return tags_dict

    def load_cachetext(self):
        read_tsv = csv.reader(open(self.args.valcaption_file, 'r'), delimiter="\t")
        caption_dict = {}
        if 'tag' in self.args.caption_type:
            tags_dict = self.load_tags()
        if self.args.caption_type == 'vinvl_tag':
            for row in read_tsv:
                if int(row[0]) not in caption_dict:
                    caption_dict[int(row[0])] = [
                        row[1].split('caption": "')[1].split('", "conf"')[0] + '. ' + tags_dict[int(row[0])]]
                else:
                    caption_dict[int(row[0])].append(
                        row[1].split('caption": "')[1].split('", "conf"')[0] + '. ' + tags_dict[int(row[0])])
        else:
            for row in read_tsv:
                if int(row[0]) not in caption_dict:
                    caption_dict[int(row[0])] = [row[1].split('caption": "')[1].split('", "conf"')[0]]
                else:
                    caption_dict[int(row[0])].append(row[1].split('caption": "')[1].split('", "conf"')[0])
        return caption_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apikey_file', type=str, default="", help='api key; https://openai.com/api/')
    parser.add_argument('--apikey', type=str, default="", help='api key; https://openai.com/api/')
    parser.add_argument('--engine', type=str, default='davinci', help='api engine; https://openai.com/api/')
    parser.add_argument('--engine_name', type=str, default='text-davinci-003', help='api engine; https://openai.com/api/')
    parser.add_argument('--caption_type', type=str, default='vinvl_tag', help='vinvl_tag, vinvl, vinvl_sg, vinvl_ocr')
    parser.add_argument('--n_shot', type=int, default=16, help="number of shots")
    parser.add_argument('--n_ensemble', type=int, default=1, help="number of ensemble")
    parser.add_argument('--rounds', type=int, default=3, help="number of interactive rounds")
    parser.add_argument('--image_id', type=int, default=-1, help="selected image id pick example only")
    parser.add_argument('--iterative_strategy', type=str, default="caption", help="caption or sg")
    parser.add_argument('--similarity_metric', type=str, default='imagequestion', help="random/question/imagequestion")
    parser.add_argument('--valcaption_file', type=str, default='input_text/vinvl_caption/VinVL_base_val2014.tsv')
    parser.add_argument('--tag_path', type=str, default='input_text/coco_caption_pred_tags')
    parser.add_argument('--concept_caption_path', type=str, default='scene_graph_coco17_caption')
    parser.add_argument('--sg_path', type=str, default='')
    parser.add_argument('--coco_path', type=str, default='coco_annotations')
    parser.add_argument('--similarity_path', type=str, default='coco_clip_new')
    parser.add_argument('--output_path', type=str, default='output')
    parser.add_argument('--llama_path', type=str, default='/')
    parser.add_argument('--use_blip2', action='store_true')
    parser.add_argument('--choice_only', action='store_true')
    parser.add_argument('--chain_of_thoughts', action='store_true')
    parser.add_argument('--with_six_gpus', action='store_true')
    parser.add_argument('--with_one_gpu', action='store_true')
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--random_attend', action='store_true')
    parser.add_argument('--oracle_attend', action='store_true')
    parser.add_argument('--random_caption', action='store_true')
    parser.add_argument('--remove_caption', action='store_true')
    parser.add_argument('--random_rationale', action='store_true')
    parser.add_argument('--oracle_rationale', action='store_true')
    parser.add_argument('--all_regional_captions', action='store_true')
    parser.add_argument('--use_attributes_to_see', action='store_true')
    parser.add_argument('--use_caption_to_see', action='store_true')
    parser.add_argument('--pick_example_mode', action='store_true')
    parser.add_argument('--pick_example_with_question_mode', action='store_true')
    parser.add_argument('--train_sim_metric', type=str, default='rationale')
    parser.add_argument('--train_sim_file', type=str, default='')
    parser.add_argument('--val_sim_file', type=str, default='')
    parser.add_argument('--verify_threshold', type=float, default=0.0)
    parser.add_argument('--start', type=float, default=0.0, help="start point in validation set (0.0-1.0)")
    parser.add_argument('--end', type=float, default=1.0, help="end point in validation set (0.0-1.0)")
    parser.add_argument('--with_clip_verify', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--ablation_visual', action='store_true')
    parser.add_argument('--ablation_reason', action='store_true')
    parser.add_argument('--use_v100', action='store_true')
    parser.add_argument('--local_rank', required=False, type=int, help='used by dist launchers')
    parser.add_argument('--raw_image_dir', type=str, default="/path/to/your/coco")
    parser.add_argument('--with_blip2_api', action='store_true')
    parser.add_argument('--set_name', type=str, default='aokvqa')
    args = parser.parse_args()

    if args.apikey_file != "":
        apikey_list = open(args.apikey_file).readlines()
        apikey_list = [line.strip() for line in apikey_list]
    else:
        apikey_list = [args.apikey]

    aokvqa = VisualCOT_AOKVQA(args, apikey_list)

    ## main inference
    #with torch.cuda.amp.autocast(dtype=torch.float):
    answers, full_answers = aokvqa.inference(save_every_step=args.engine in ['ada', 'babbage', 'curie', 'davinci', 'gpt3',
                                                                   'chat', 'codex', 'instruct'] or
                                               args.pick_example_mode)

    # prediction = {}
    acc = 0.
    # for answer in answers:
    #     prediction[answer[0]] = [answer[1], answer[2]]
    #     acc += float(answer[3])

    format_prediction = []
    for answer in answers:
        if args.chain_of_thoughts:
            format_prediction.append({"answer": answer[1], "question_id": answer[0].split('<->')[1],
                                      "thoughts": answer[5]})
        else:
            format_prediction.append({"answer": answer[1], "question_id": answer[0].split('<->')[1]})

    print(acc * 100. / len(answers), len(answers))
    acc = acc * 100. / len(answers)

    ## if save final predictions
    os.system("mkdir -p %s" % args.output_path)
    os.system("mkdir -p %s/prompt_answer" % args.output_path)
    os.system("mkdir -p %s/format_answer" % args.output_path)
    output_name = 'VisualCOT_%s_n%d_repeat%d_%s_%f.json' % (
    args.caption_type, args.n_shot, args.n_ensemble, args.similarity_metric, acc)
    json.dump(full_answers, open("%s/prompt_answer/%s" % (args.output_path, output_name), 'w'))
    json.dump(format_prediction, open("%s/format_answer/%s" % (args.output_path, output_name), 'w'))

if __name__ == '__main__':
    main()
