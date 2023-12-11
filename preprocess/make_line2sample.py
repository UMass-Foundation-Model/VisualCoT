import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True, help='path to input')
parser.add_argument('--output', type=str, required=True, help='path to output')
args = parser.parse_args()

inputs = json.load(open(args.input))
output_dict = {}
if 'questions' in inputs:
    # VQA v2
    for idx, q in enumerate(inputs['questions']):
        val = "<->".join([str(q['image_id']), str(q['question_id'])])
        output_dict[str(idx)] = val
else:
    # A-OKVQA
    for idx, q in enumerate(inputs):
        val = "<->".join([str(q['image_id']), str(q['question_id'])])
        output_dict[str(idx)] = val
json.dump(output_dict, open(args.output, "w"))
