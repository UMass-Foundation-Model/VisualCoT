import os
import argparse
import json
import torch
import openai
from transformers import CLIPTokenizer, CLIPTextModel
import pickle

def load_anno(coco_caption_file,answer_anno_file,question_anno_file):
    if coco_caption_file is not None:
        coco_caption = json.load(open(coco_caption_file,'r'))
        if type(coco_caption)==type({}): coco_caption = coco_caption['annotations']
    answer_anno = json.load(open(answer_anno_file,'r'))
    question_anno = json.load(open(question_anno_file,'r'))

    caption_dict = {}
    if coco_caption_file is not None:
        for sample in coco_caption:
            if sample['image_id'] not in caption_dict:
                caption_dict[sample['image_id']] = [sample['caption']]
            else:
                caption_dict[sample['image_id']].append(sample['caption'])
    answer_dict = {}
    for sample in answer_anno["annotations"]:
        if str(sample['image_id'])+'<->'+str(sample['question_id']) not in answer_dict:
            answers = [ans['raw_answer'] for ans in sample['answers']]
            answer_dict[str(sample['image_id'])+'<->'+str(sample['question_id'])] = answers

    question_dict = {}
    for sample in question_anno['questions']:
        if str(sample['image_id'])+'<->'+str(sample['question_id']) not in question_dict:
            question_dict[str(sample['image_id'])+'<->'+str(sample['question_id'])] = sample['question']

    rationales_dict = {}
    return caption_dict,answer_dict,question_dict,rationales_dict

class OKVQA:
    def __init__(self, args):
        self.args = args
        self.chain_of_thoughts = args.chain_of_thoughts
        _,self.answer_dict,self.question_dict,self.rationale_dict = \
            load_anno(None, '%s/mscoco_val2014_annotations.json'%args.coco_path, \
                '%s/OpenEnded_mscoco_val2014_questions.json'%args.coco_path)
        self.val_keys = list(self.question_dict.keys())
        self.val_keys = self.val_keys[int(args.start*len(self.val_keys)):int(args.end*len(self.val_keys))]

        self.traincontext_caption_dict,self.traincontext_answer_dict,\
        self.traincontext_question_dict,self.traincontext_rationale_dict = \
            load_anno('%s/captions_train2014.json'%args.coco_path, \
                '%s/mscoco_train2014_annotations.json'%args.coco_path, \
                '%s/OpenEnded_mscoco_train2014_questions.json'%args.coco_path)
        self.train_keys = list(self.traincontext_answer_dict.keys())

        self.sg_dir = os.path.join(self.args.sg_path, "scene_graph_coco17")
        self.sg_attr_dir = os.path.join(self.args.sg_path, "scene_graph_coco17_attr")
        self.sg_cap_dir = os.path.join(self.args.sg_path, "scene_graph_coco17_caption")

    def get_related_obj_dict(self, key, metric, model=None, processor=None):
        img_context_key = int(key.split('<->')[0])
        context_scene_graph = json.load(open(os.path.join(self.sg_dir, str(img_context_key).zfill(12) + ".json")))
        context_scene_graph_attr = json.load(
            open(os.path.join(self.sg_attr_dir, str(img_context_key).zfill(12) + ".json")))

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

        related_obj_dict = {}
        if 'rationale' in metric:
            rationale = self.traincontext_rationale_dict[key]
            for obj in obj_list:
                for r in rationale:
                    if obj in r:
                        if obj not in related_obj_dict:
                            related_obj_dict[obj] = 1
                        else:
                            related_obj_dict[obj] += 1
        elif 'answer' in metric:
            with torch.no_grad():
                answer_list = self.traincontext_answer_dict[key]
                inputs = processor(text=answer_list, return_tensors="pt", padding=True)
                inputs = {k: v.cuda() for k, v in inputs.items()}
                outputs = model(**inputs)
                ans_text_emb = outputs['pooler_output'].mean(dim=0).unsqueeze(dim=0)

                inputs = processor(text=obj_list, return_tensors="pt", padding=True)
                inputs = {k: v.cuda() for k, v in inputs.items()}
                outputs = model(**inputs)
                cand_text_emb = outputs['pooler_output']

                ans_text_emb /= ans_text_emb.norm(dim=-1, keepdim=True)
                cand_text_emb /= cand_text_emb.norm(dim=-1, keepdim=True)

                sim_cands = cand_text_emb @ ans_text_emb.T
        for idx, obj_name in enumerate(obj_list):
                related_obj_dict[obj_name] = sim_cands[idx, 0].detach().cpu().item()
        return obj_list, conf_list, related_obj_dict

    def show_object_example(self):
        metric_list = ['rationale', 'answer', 'question']        
        prompt = 'Please select the object most related to the question.\n===\n'
        metric = metric_list[1]
        
        out_train_fn = "./input_text/scene_graph_text/train_object_select_okvqa.pk"
        
        if 'answer' in metric:         
            model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16")
            processor = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
            model = model.cuda()
        else:
            model, processor = None, None

        out_object_sim_dict = {}

        for pid, img_ques in enumerate(self.train_keys):
            obj_list, conf_list, rel_obj_dict = self.get_related_obj_dict(img_ques, metric, model, processor)
            rel_obj = [k for k, v in sorted(rel_obj_dict.items(), key=lambda item: item[1], reverse=True)]
            
            prompt += 'Question: %s\n===\nOptions:\n' % (self.traincontext_question_dict[img_ques])
            candidate_list = [cls for cls, conf in sorted(zip(obj_list, conf_list), key=lambda item: item[1], reverse=True)]
            if rel_obj[0] not in candidate_list:
                candidate_list.append(rel_obj[0])
            for oi, obj in enumerate(candidate_list):
                prompt += "%s: %s\n" % (chr(ord("A")+oi), obj)
            prompt += "The most related option is %s: %s\n\n===\n" % (chr(ord("A")+candidate_list.index(rel_obj[0])), rel_obj[0])
            prompt += "The most related option %s\n\n===\n" % (rel_obj[0])
            if pid % 100 ==0:
                print("%d/%d"%(pid, len(self.train_keys))) 
            out_object_sim_dict[img_ques] =   rel_obj_dict 
        with open(out_train_fn, "wb") as fh:
            pickle.dump(out_object_sim_dict, fh)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apikey', type=str, required=True, help='api key; https://openai.com/api/')
    parser.add_argument('--engine', type=str, default='davinci', help='api engine; https://openai.com/api/')
    parser.add_argument('--caption_type', type=str, default='vinvl_tag', help='vinvl_tag, vinvl, vinvl_sg')
    parser.add_argument('--n_shot', type=int, default=16, help="number of shots")
    parser.add_argument('--n_ensemble', type=int, default=1, help="number of ensemble")
    parser.add_argument('--rounds', type=int, default=3, help="number of interactive rounds")
    parser.add_argument('--iterative_strategy', type=str, default="caption", help="caption or sg")
    parser.add_argument('--similarity_metric', type=str, default='imagequestion', help="random/question/imagequestion")
    parser.add_argument('--valcaption_file', type=str, default='input_text/vinvl_caption/VinVL_base_val2014.tsv')
    parser.add_argument('--tag_path', type=str, default='input_text/coco_caption_pred_tags')
    parser.add_argument('--sg_path', type=str, default='')
    parser.add_argument('--coco_path', type=str, default='coco_annotations')
    parser.add_argument('--similarity_path', type=str, default='coco_clip_new')
    parser.add_argument('--output_path', type=str, default='output')
    parser.add_argument('--chain_of_thoughts', action='store_true')
    parser.add_argument('--with_six_gpus', action='store_true')
    parser.add_argument('--start', type=float, default=0.0, help="start point in validation set (0.0-1.0)")
    parser.add_argument('--end', type=float, default=1.0, help="end point in validation set (0.0-1.0)")
    args = parser.parse_args()

    openai.api_key = args.apikey

    okvqa = OKVQA(args)

    okvqa.show_object_example()

if __name__ == '__main__':
    main()
