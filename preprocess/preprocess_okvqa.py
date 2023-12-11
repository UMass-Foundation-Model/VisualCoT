import pdb
import json
import os

BASE_PATH = "/PATH/TO/VisualCOT"

def reorganize_captions():
	cap_path = "/pathto/val_refine.json"
	sg_attr_dir = f"{BASE_PATH}/input_text/scene_graph_text/scene_graph_coco17_attr"
	out_dir = f"{BASE_PATH}/input_text/scene_graph_text/scene_graph_coco14_caption_ok"
	
	if not os.path.isdir(out_dir):
		os.mkdir(out_dir)

	cap_dict_list = json.load(open(cap_path))	
	
	def cap_dict_to_img(cap_dict_list):
		cap_dict_img = {}
		for idx, rgn_info in enumerate(cap_dict_list):
			img_id = rgn_info["image_id"].split("[")[0]
			if img_id not in cap_dict_img:
				cap_dict_img[img_id] = []
			cap_dict_img[img_id].append(rgn_info)
		return cap_dict_img 

	cap_dict_img = cap_dict_to_img(cap_dict_list)
	
	for img_id_str, rgn_list in cap_dict_img.items():

		out_img_path = os.path.join(out_dir, img_id_str.zfill(12)+".json")

		def rgn2dict(rgn_list):
			rgn_dict = {}
			for rgn in rgn_list:
				rgn_box_id = rgn["image_id"].split("[")[1]
				rgn_box_id = "[" + rgn_box_id
				if rgn_box_id not in rgn_dict:
					rgn_dict[rgn_box_id] = [rgn]
				else:
					rgn_dict[rgn_box_id].append(rgn)
			return rgn_dict

		rgn_dict = rgn2dict(rgn_list)

		scene_graph_attr = json.load(open(os.path.join(sg_attr_dir, img_id_str.zfill(12) + ".json")))
		cap_list = []
		
		for idx, rgn in enumerate(scene_graph_attr[0]):
			rgn_id = str(rgn["rect"])
			if rgn_id in rgn_dict:
				if len(rgn_dict[rgn_id])==1:
					cap_list.append(rgn_dict[rgn_id][0]["caption"])
				else:
					find_valid_flag = False
					for tmp_idx in range(len(rgn_dict[rgn_id])):
						tmp_dict = rgn_dict[rgn_id][tmp_idx]
						if rgn["class"] in tmp_dict["concept"]:
							cap_list.append(rgn_dict[rgn_id][tmp_idx]["caption"])
							find_valid_flag = True
							break
					#assert find_valid_flag
					if not find_valid_flag:
						import pdb; pdb.set_trace()
			else:
				attr_str = ""
				if len(rgn["attr_conf"])>0:
					val = max(rgn["attr_conf"])
					idx = rgn["attr_conf"].index(val)
					attr_str = "%s "%(rgn["attr"][idx])
				fake_cap = "%s %s"%(attr_str, rgn["class"]) 
				fake_cap = fake_cap.strip()
				cap_list.append(fake_cap)
				print(rgn)
				print(fake_cap)
		with open(out_img_path, "w") as fh:
			json.dump(cap_list, fh)
		#import pdb; pdb.set_trace()

if __name__=="__main__":
	reorganize_captions()