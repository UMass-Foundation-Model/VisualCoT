import os
import argparse
import numpy as np
import json
import torch
from PIL import Image
from main_aokvqa import VisualCOT_AOKVQA

class VisualCOT(VisualCOT_AOKVQA):
    def __init__(self, args, apikey_list):
        super().__init__(args, apikey_list)
        self.train_ok_keys = list(self.traincontext_ok_answer_dict.keys())

    def find_image(self, img_key):
        split = "test" if self.args.test_only else "val"
        img_full_path = os.path.join(self.raw_image_dir, "COCO_%s2014_%012d.jpg" % (split, img_key))
        print(img_full_path)
        return Image.open(img_full_path).convert("RGB")

    def load_dataset(self, args):
        test_name = "test" if args.test_only else "val"
        self.raw_image_dir = os.path.join(self.args.raw_image_dir, "%s2014" % test_name)
        _, self.answer_dict, self.question_dict = \
            self.load_ok_anno(None, f'%s/mscoco_{test_name}2014_annotations.json' % args.coco_path, \
                         f'%s/OpenEnded_mscoco_{test_name}2014_questions.json' % args.coco_path)
        self.val_keys = list(self.question_dict.keys())
        self.val_keys = self.val_keys[int(args.start * len(self.val_keys)):int(args.end * len(self.val_keys))]

        ## load cached image representation (Coco caption & Tags)
        self.inputtext_dict = self.load_cachetext()

        if self.args.with_ok_context:
            self.traincontext_caption_dict, self.traincontext_answer_dict, \
            self.traincontext_question_dict = \
                self.load_ok_anno(f"%s/captions_train2017.json" % args.coco_path, \
                             f'%s/mscoco_train2014_annotations.json' % args.coco_path, \
                             f'%s/OpenEnded_mscoco_train2014_questions.json' % args.coco_path)
            self.traincontext_ok_answer_dict = self.traincontext_answer_dict
            self.traincontext_ok_question_dict = self.traincontext_question_dict
            # without chain_of_thoughts for ok context
            assert not self.args.chain_of_thoughts
        else:
            self.traincontext_caption_dict, self.traincontext_answer_dict, \
            self.traincontext_question_dict, self.traincontext_rationale_dict, \
            self.traincontext_choices_dict = \
                self.load_aok_anno('%s/captions_train2017.json' % args.coco_path, \
                              '%s/aokvqa_v1p0_train.json' % args.coco_path, \
                              '%s/aokvqa_v1p0_train.json' % args.coco_path, choice_only=args.choice_only)

            _, self.traincontext_ok_answer_dict, \
            self.traincontext_ok_question_dict = \
                self.load_ok_anno(None, f'%s/mscoco_train2014_annotations.json' % args.coco_path, \
                             f'%s/OpenEnded_mscoco_train2014_questions.json' % args.coco_path)
        if args.caption_type == 'vinvl_ocr':
            self.load_ocr(os.path.join(self.args.sg_path, "coco14_ocr_train.json"),
                          os.path.join(self.args.sg_path, f"coco14_ocr_{test_name}.json"),
                          os.path.join(self.args.sg_path, "scene_graph_coco17_attr"))
        self.sg_dir = os.path.join(self.args.sg_path, "scene_graph_coco17")
        self.sg_attr_dir = os.path.join(self.args.sg_path, "scene_graph_coco17_attr")
        self.sg_cap_dir = os.path.join(self.args.sg_path, self.args.concept_caption_path)

        self.train_keys = list(self.traincontext_answer_dict.keys())
        self.train_interactive_keys = list(self.traincontext_ok_answer_dict.keys())
        self.traincontext_interactive_answer_dict = self.traincontext_ok_answer_dict
        self.traincontext_interactive_question_dict = self.traincontext_ok_question_dict

    def load_ok_anno(self, coco_caption_file, answer_anno_file, question_anno_file):
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
        for sample in answer_anno['annotations']:
            if str(sample['image_id']) + '<->' + str(sample['question_id']) not in answer_dict:
                answer_dict[str(sample['image_id']) + '<->' + str(sample['question_id'])] = [x['answer'] for x in
                                                                                             sample['answers']]

        question_dict = {}
        for sample in question_anno['questions']:
            if str(sample['image_id']) + '<->' + str(sample['question_id']) not in question_dict:
                question_dict[str(sample['image_id']) + '<->' + str(sample['question_id'])] = sample['question']
        return caption_dict, answer_dict, question_dict

    def load_aok_anno(self, coco_caption_file, answer_anno_file, question_anno_file, choice_only=False):
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
                        assert False
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

    def get_context_keys(self, key, metric, n):
        if not self.args.with_ok_context:
            if metric == 'question':
                lineid = self.valkey2idx[key]
                similarity = np.matmul(self.train_feature, self.val_feature[lineid, :])
                index = similarity.argsort()[-n:][::-1]
                return [self.train_idx[str(x)] for x in index]
            elif metric == 'imagequestion':
                ## combined with Q-similairty (image+question)
                lineid = self.valkey2idx[key]
                question_similarity = np.matmul(self.train_feature, self.val_feature[lineid, :])
                ## end of Q-similairty
                similarity = question_similarity + np.matmul(self.image_train_feature, self.image_val_feature[lineid, :])
                index = similarity.argsort()[-n:][::-1]
                return [self.train_idx[str(x)] for x in index]
            else:
                return None
        else:
            if metric == 'question':
                lineid = self.valkey2idx[key]
                similarity = np.matmul(self.train_ok_feature, self.val_feature[lineid, :])
                index = similarity.argsort()[-n:][::-1]
                return [self.train_ok_idx[str(x)] for x in index]
            elif metric == 'imagequestion':
                ## combined with Q-similairty (image+question)
                lineid = self.valkey2idx[key]
                question_similarity = np.matmul(self.train_ok_feature, self.val_feature[lineid, :])
                ## end of Q-similairty
                similarity = question_similarity + np.matmul(self.image_train_ok_feature, self.image_val_feature[lineid, :])
                index = similarity.argsort()[-n:][::-1]
                return [self.train_ok_idx[str(x)] for x in index]
            else:
                return None

    def get_interactive_context_keys(self, key, metric, n):
        if metric == 'question':
            assert False
        elif metric == 'imagequestion':
            ## combined with Q-similairty (image+question)
            lineid = self.valkey2idx[key]
            question_similarity = np.matmul(self.train_ok_feature, self.val_feature[lineid, :])
            ## end of Q-similairty
            similarity = question_similarity + np.matmul(self.image_train_ok_feature, self.image_val_feature[lineid, :])
            similarity = similarity.argsort()
            idx_list = []
            rel_obj_list = []
            for i in range(len(similarity)):
                context_key = self.train_ok_idx[str(similarity[-1 - i])]
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
        val_idx = json.load(open('%s/okvqa_qa_line2sample_idx_%s2014.json' % (self.args.similarity_path, split), 'r'))
        self.valkey2idx = {}
        for ii in val_idx:
            self.valkey2idx[val_idx[ii]] = int(ii)
        if self.args.similarity_metric == 'question':
            self.train_feature = np.load(
                '%s/coco_clip_vitb16_train2017_aokvqa_question.npy' % self.args.similarity_path)
            self.train_ok_feature = np.load(
                '%s/coco_clip_vitb16_train2014_okvqa_question.npy' % self.args.similarity_path)
            self.val_feature = np.load(
                '%s/coco_clip_vitb16_%s2014_okvqa_question.npy' % (self.args.similarity_path, split))
            self.train_idx = json.load(
                open('%s/aokvqa_qa_line2sample_idx_train2017.json' % self.args.similarity_path, 'r'))
            self.train_ok_idx = json.load(
                open('%s/okvqa_qa_line2sample_idx_train2014.json' % self.args.similarity_path, 'r'))
        elif self.args.similarity_metric == 'imagequestion':
            self.train_feature = np.load(
                '%s/coco_clip_vitb16_train2017_aokvqa_question.npy' % self.args.similarity_path)
            self.train_ok_feature = np.load(
                '%s/coco_clip_vitb16_train2014_okvqa_question.npy' % self.args.similarity_path)
            self.val_feature = np.load(
                '%s/coco_clip_vitb16_%s2014_okvqa_question.npy' % (self.args.similarity_path, split))
            self.train_idx = json.load(
                open('%s/aokvqa_qa_line2sample_idx_train2017.json' % self.args.similarity_path, 'r'))
            self.train_ok_idx = json.load(
                open('%s/okvqa_qa_line2sample_idx_train2014.json' % self.args.similarity_path, 'r'))
            self.image_train_feature = np.load(
                '%s/coco_clip_vitb16_train2017_aokvqa_convertedidx_image.npy' % self.args.similarity_path)
            self.image_train_ok_feature = np.load(
                '%s/coco_clip_vitb16_train2014_okvqa_convertedidx_image.npy' % self.args.similarity_path)
            self.image_val_feature = np.load(
                '%s/coco_clip_vitb16_%s2014_okvqa_convertedidx_image.npy' % (self.args.similarity_path, split))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apikey_file', type=str, default="", help='api key; https://openai.com/api/')
    parser.add_argument('--apikey', type=str, default="", help='api key; https://openai.com/api/')
    parser.add_argument('--engine', type=str, default='davinci', help='api engine; https://openai.com/api/')
    parser.add_argument('--engine_name', type=str, default='text-davinci-003', help='api engine name')
    parser.add_argument('--caption_type', type=str, default='vinvl_tag', help='vinvl_tag, vinvl, vinvl_sg, vinvl_ocr')
    parser.add_argument('--n_shot', type=int, default=16, help="number of shots")
    parser.add_argument('--n_ensemble', type=int, default=1, help="number of ensemble")
    parser.add_argument('--rounds', type=int, default=3, help="number of interactive rounds")
    parser.add_argument('--iterative_strategy', type=str, default="caption", help="caption or sg")
    parser.add_argument('--similarity_metric', type=str, default='imagequestion', help="random/question/imagequestion")
    parser.add_argument('--valcaption_file', type=str, default='input_text/vinvl_caption/VinVL_base_val2014.tsv')
    parser.add_argument('--tag_path', type=str, default='input_text/coco_caption_pred_tags')
    parser.add_argument('--concept_caption_path', type=str, default="scene_graph_coco14_caption_ok")
    parser.add_argument('--sg_path', type=str, default='')
    parser.add_argument('--coco_path', type=str, default='coco_annotations')
    parser.add_argument('--similarity_path', type=str, default='coco_clip_new')
    parser.add_argument('--output_path', type=str, default='output')
    parser.add_argument('--use_blip2', action='store_true')
    parser.add_argument('--choice_only', action='store_true')
    parser.add_argument('--chain_of_thoughts', action='store_true')
    parser.add_argument('--all_regional_captions', action='store_true')
    parser.add_argument('--use_attributes_to_see', action='store_true')
    parser.add_argument('--with_six_gpus', action='store_true')
    parser.add_argument('--with_one_gpu', action='store_true')
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--random_attend', action='store_true')
    parser.add_argument('--oracle_attend', action='store_true')
    parser.add_argument('--random_caption', action='store_true')
    parser.add_argument('--remove_caption', action='store_true')
    parser.add_argument('--random_rationale', action='store_true')
    parser.add_argument('--oracle_rationale', action='store_true')
    parser.add_argument('--llama_path', type=str, default='/')
    parser.add_argument('--train_sim_metric', type=str, default='rationale')
    parser.add_argument('--train_sim_file', type=str, default='')
    parser.add_argument('--val_sim_file', type=str, default='')
    parser.add_argument('--verify_threshold', type=float, default=0.0)
    parser.add_argument('--start', type=float, default=0.0, help="start point in validation set (0.0-1.0)")
    parser.add_argument('--end', type=float, default=1.0, help="end point in validation set (0.0-1.0)")
    parser.add_argument('--with_clip_verify', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--with_ok_context', action='store_true')
    parser.add_argument('--ablation_visual', action='store_true')
    parser.add_argument('--ablation_reason', action='store_true')
    parser.add_argument('--use_v100', action='store_true')
    parser.add_argument('--local_rank', required=False, type=int, help='used by dist launchers')
    parser.add_argument('--raw_image_dir', type=str, default="/path/to/your/coco")
    parser.add_argument('--with_blip2_api', action='store_true')
    parser.add_argument('--set_name', type=str, default='okvqa')
    args = parser.parse_args()

    if args.apikey_file != "":
        apikey_list = open(args.apikey_file).readlines()
        apikey_list = [line.strip() for line in apikey_list]
    else:
        apikey_list = [args.apikey]

    okvqa = VisualCOT(args, apikey_list=apikey_list)

    ## main inference
    #with torch.cuda.amp.autocast(dtype=torch.float):
    answers = okvqa.inference(save_every_step=args.engine in ['ada', 'babbage', 'curie', 'davinci', 'chat', 'codex', 'instruct', 'gpt3'])

    prediction = {}
    acc = 0.
    for answer in answers:
        prediction[answer[0]] = [answer[1], answer[2]]
        acc += float(answer[3])

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
    json.dump(prediction, open("%s/prompt_answer/%s" % (args.output_path, output_name), 'w'))
    json.dump(format_prediction, open("%s/format_answer/%s" % (args.output_path, output_name), 'w'))


if __name__ == '__main__':
    main()
