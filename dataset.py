import os
import json

from utils import get_first_number_form_str, create_dict_from_files, convert_json_tables_to_html, remove_special_formats

class ChemTableDataset:
    def __init__(self, item_len=500000, source_path="data/"):
        folders = ["json", "img", "sub_img"]
        dicts = {}
        for folder in folders:
            files = os.listdir(os.path.join(source_path, folder))
            dicts[folder] = create_dict_from_files(files, source_path, folder)

        self.data_list = []
        available_ids = sorted(dicts["json"].keys())
        item_ids = available_ids[:item_len] if len(available_ids) > item_len else available_ids
        
        for i in item_ids:
            with open(dicts["json"][i][0], 'r', encoding='utf-8') as f:
                data = json.load(f)
            smiles_list = []
            current_sub_imgs = []
            reaction_list = data["data"]["reactions"]
            table_list = data["data"]["tables"]
            substance_list = data["data"]["substances"]

            title_list = data["data"].get("title", [])
            title_text_list = [remove_special_formats(title_item["text"]) for title_item in title_list] if title_list else []
            
            annotations_list = data["data"].get("annotations", [])
            annotations_text_list = [remove_special_formats(anno_item["text"]) for anno_item in annotations_list] if annotations_list else []

            if i in dicts["sub_img"]:
                current_sub_imgs = dicts["sub_img"][i]
            for reaction in reaction_list:
                if i not in dicts.get("sub_img", {}):
                    continue

                for part in reaction["reactants"] + reaction["conditions"] + reaction["products"]:
                    sub_image_path = f"{source_path}sub_img\\{part['id']}.png"
                    if sub_image_path in current_sub_imgs and part.get("maps"):
                        if len(part["maps"]) > 1:
                            continue
                        smiles_gt = part["maps"][0]["smiles"]
                        if smiles_gt == "":
                            continue
                        smiles_list.append({
                            "smiles_id": part["id"],
                            "smiles_image_path": sub_image_path,
                            "smiles_gt": smiles_gt
                        })
            for cells in table_list[0]["data"]:
                if len(cells["maps"]) != 0:
                    sub_image_path = f"{source_path}sub_img\\{cells['id']}.png"
                    if sub_image_path in current_sub_imgs:
                        smiles_gt = cells["maps"][0]["smiles"]
                        if smiles_gt == "":
                            continue
                        smiles_list.append({
                            "smiles_id": cells["id"],
                            "smiles_image_path": sub_image_path,
                            "smiles_gt": smiles_gt
                        })
            for substance in substance_list:
                if len(substance["maps"]) != 0:
                    sub_image_path = f"{source_path}sub_img\\{substance['id']}.png"
                    if sub_image_path in current_sub_imgs:
                        smiles_gt = substance["maps"][0]["smiles"]
                        if smiles_gt == "":
                            continue
                        smiles_list.append({
                            "smiles_id": substance["id"],
                            "smiles_image_path": sub_image_path,
                            "smiles_gt": smiles_gt
                        })
            html_list = convert_json_tables_to_html(dicts["json"][i][0])
            
            item_json = {
                "id": i,
                "clear_table_html": html_list[0],
                "image_path": dicts["img"][i][0],
                "smiles": smiles_list,
                "title": title_text_list,
                "annotations": annotations_text_list,
                "reaction_list": reaction_list,
            }
            self.data_list.append(item_json)

    def getDataList(self):
        return self.data_list
