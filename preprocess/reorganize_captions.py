import json
import os
import argparse

def find_files(dir, is14=False):
    files = [f.split(".")[0] for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
    if is14:
        files = [f.split("_")[-1] for f in files]
    return files

parser = argparse.ArgumentParser()
parser.add_argument('--coco14root', type=str, required=True, help='path to coco14')
parser.add_argument('--coco17root', type=str, required=True, help='path to coco17')
parser.add_argument('--caption14train', type=str, required=True, help='path to 14 caption train')
parser.add_argument('--caption14val', type=str, required=True, help='path to 14 caption val')
parser.add_argument('--caption17train', type=str, required=True, help='output path to 14 caption train')
parser.add_argument('--caption17val', type=str, required=True, help='output path to 14 caption val')
args = parser.parse_args()

train17dir = os.path.join(args.coco17root, "train2017")
val17dir = os.path.join(args.coco17root, "val2017")

train14dir = os.path.join(args.coco14root, "train2014")
val14dir = os.path.join(args.coco14root, "val2014")

split14 = [find_files(train14dir, is14=True), find_files(val14dir, is14=True)]
split17 = [find_files(train17dir, is14=False), find_files(val17dir, is14=False)]

caption14train = json.load(open(args.caption14train))['annotations']
caption14val = json.load(open(args.caption14val))['annotations']
caption14 = caption14train + caption14val
caption14_dict = {}
for c in caption14:
    if c['image_id'] not in caption14_dict:
        caption14_dict[c['image_id']] = [c['caption']]
    else:
        caption14_dict[c['image_id']].append(c['caption'])

caption17train = {
    "annotations":  []
}
caption17val = {
    "annotations":  []
}

for iid in split17[0]:
    iid = int(iid)
    captions = caption14_dict[iid]
    for cp in captions:
        caption17train["annotations"].append(
                {"image_id": iid,
                 "caption": cp}
        )
for iid in split17[1]:
    iid = int(iid)
    captions = caption14_dict[iid]
    for cp in captions:
        caption17val["annotations"].append(
                {"image_id": iid,
                 "caption": cp}
        )

json.dump(caption17train, open(args.caption17train, "w"))
json.dump(caption17val, open(args.caption17val, "w"))
