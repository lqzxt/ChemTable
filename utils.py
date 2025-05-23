import imghdr
import json
import os
import re
import base64
import io
import warnings
from template import qa_answer_eval
from LLM import call_LLM

from bs4 import BeautifulSoup
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image


def remove_special_formats(input_str):
    pattern1 = r'\\textbf\{(.*?)\}'
    result = re.sub(pattern1, r'\1', input_str)

    pattern2 = r'\\refiden\{(.*?)\}'
    result = re.sub(pattern2, r'\1', result)

    pattern2 = r'\\refmark\{(.*?)\}'
    result = re.sub(pattern2, r'\1', result)

    pattern2 = r'\\textit\{(.*?)\}'
    result = re.sub(pattern2, r'\1', result)

    pattern2 = r'\\iden\{(.*?)\}'
    result = re.sub(pattern2, r'\1', result)

    pattern2 = r'\\color\{[^}]+\}\{(.*?)\}'
    result = re.sub(pattern2, r'\1', result)

    pattern2 = r'\\mark\{(.*?)\}'
    result = re.sub(pattern2, r'\1', result)

    result = ' '.join(result.split())
    result = result.replace("（", "(").replace("）", ")")
    return result


def generate_html_table(cells):
    max_row = max(cell['end_row'] for cell in cells) + 1
    max_col = max(cell['end_col'] for cell in cells) + 1
    table = [[None for _ in range(max_col)] for _ in range(max_row)]
    for cell in cells:
        start_row = cell['start_row']
        end_row = cell['end_row']
        start_col = cell['start_col']
        end_col = cell['end_col']
        content = cell['content']
        rowspan = end_row - start_row + 1
        colspan = end_col - start_col + 1
        for row in range(start_row, end_row + 1):
            for col in range(start_col, end_col + 1):
                if row == start_row and col == start_col:
                    table[row][col] = {
                        'rowspan': rowspan if rowspan > 1 else None,
                        'colspan': colspan if colspan > 1 else None,
                        'tex': content
                    }
                else:
                    table[row][col] = 'merged'
    html_table = '<table>'
    for row in table:
        html_table += '<tr>'
        for cell in row:
            if cell is None:
                html_table += '<td></td>'
            elif cell == 'merged':
                continue
            else:
                cell_html = '<td'
                if cell['rowspan']:
                    cell_html += f' rowspan="{cell["rowspan"]}"'
                if cell['colspan']:
                    cell_html += f' colspan="{cell["colspan"]}"'
                cell_html += f">{cell['tex']}</td>"
                html_table += cell_html
        html_table += '</tr>'
    html_table += '</table>'
    return html_table


def convert_json_tables_to_html(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    html_list = []
    for table in data["data"]["tables"]:
        table_cells = []
        for cell in table["data"]:
            content = cell["text"]
            if len(cell["maps"]) != 0:
                content = cell["maps"][0]["smiles"]

            temp = {
                "start_row": int(cell["start_row"]) - 1,
                "end_row": int(cell["end_row"]) - 1,
                "start_col": int(cell["start_col"]) - 1,
                "end_col": int(cell["end_col"]) - 1,
                "content": content
            }
            table_cells.append(temp)
        html = generate_html_table(table_cells)
        html = remove_special_formats(html)
        html_list.append(html)
    return html_list


def get_first_number_form_str(input_str):
    pattern = r'^\d+'
    match = re.search(pattern, input_str)
    if match:
        result = int(match.group())
        return result
    else:
        print("No number found at the beginning.")
        return -1


def is_same_molecule(smiles1, smiles2):
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)

        if mol1 is None and mol2 is not None:
            return False
        if mol1 is not None and mol2 is None:
            return False

        canonical_smiles1 = Chem.MolToSmiles(mol1)
        canonical_smiles2 = Chem.MolToSmiles(mol2)
    except Exception as e:
        print(e)
        return False
    return canonical_smiles1 == canonical_smiles2


def calculate_tanimoto_similarity(smiles1, smiles2):
    try:
        from rdkit import DataStructs
        from rdkit.Chem import AllChem
        
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        
        if mol1 is None or mol2 is None:
            return 0.0
        
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
        
        similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
        return similarity
    except Exception as e:
        print(f"Error calculating Tanimoto similarity: {e}")
        return 0.0


def look_molecule(smiles_list):
    mol_list = []
    try:
        for item in smiles_list:
            mol_list.append(Chem.MolFromSmiles(item))
        img = Draw.MolsToImage(mol_list, subImgSize=(600, 600))
        img.save('molecule.png')
    except Exception as e:
        print(e)
        return False


def look_html(res_path="res_TR.jsonl", output_folder="temp_html/"):
    with open(res_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    TEDS_Sum = 0.0
    TEDS_Struct_Sum = 0.0
    count = 0
    for index, line in enumerate(lines):
        data = json.loads(line)
        TEDS_Sum += data['TEDS']
        TEDS_Struct_Sum += data['TEDS_Struct']
        count += 1
        output_path = output_folder + str(data["index"]) + '.html'
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(data['pre'])
            f.write("<br><br><br>")
            f.write(data["gt"])
            f.write("<style>table, th, td {border: 1px solid black;border-collapse: collapse;}</style>")
    print(f"TEDS: {TEDS_Sum / count}, TEDS_Struct: {TEDS_Struct_Sum / count}")


def create_dict_from_files(files, source_path, folder):
    result_dict = {}
    for item in files:
        index = get_first_number_form_str(item)
        if index not in result_dict:
            result_dict[index] = []
        result_dict[index].append(os.path.join(source_path, folder, item))
    return result_dict


def encode_image(image_path):
    with Image.open(image_path) as image:
        if image.format == "PNG":
            image = image.convert("RGB")
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")
            buffer.seek(0)
            encoded_image = base64.b64encode(buffer.read()).decode('utf-8')
        else:
            with open(image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_image


def get_image_type(image_path):
    image_type = imghdr.what(image_path)
    return image_type if image_type else "jpeg"


def create_prompt(mes, image_path=None):
    if not image_path:
        return create_prompt_text(mes)
    base64_image = encode_image(image_path)
    return [{
        "role": "user",
        "content": [
            {"type": "text", "text": mes},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{get_image_type(image_path)};base64,{base64_image}"
                }
            }
        ]
    }]


def create_prompt_text(mes):
    return [{
        "role": "user",
        "content": [
            {"type": "text", "text": mes},
        ]
    }]

def extract_HTML(text):
    start_index = text.find('```html')
    end_index = text.rfind('</table>')
    if start_index != -1 and end_index != -1:
        s = text[start_index + 7:end_index + 8]
        s = s.strip()
        soup = BeautifulSoup(s, 'html.parser')
        cleaned_html = str(soup.table)
        cleaned_html = cleaned_html.replace("\n", "")
        return cleaned_html
    else:
        start_index = text.find('<table>')
        if start_index != -1 and end_index != -1:
            s = text[start_index:end_index + 8]
            s = s.strip()
            soup = BeautifulSoup(s, 'html.parser')
            cleaned_html = str(soup.table)
            cleaned_html = cleaned_html.replace("\n", "")
            return cleaned_html
        raise Exception("Parse error! Not find HTML in LLM response!")

def format_td(html):
    html = html.replace("<b>", "").replace("</b>", "")
    html = html.replace("<i>", "").replace("</i>", "")
    html = html.replace("<sup>", "").replace("</sup>", "")
    html = html.replace("<sub>", "").replace("</sub>", "")
    html = html.replace("<u>", "").replace("</u>", "")
    html = html.replace("<span>", "").replace("</span>", "")
    html = html.replace("<strong>", "").replace("</strong>", "")
    html = html.replace("<em>", "").replace("</em>", "")
    html = html.replace("\n", "")
    html = html.replace("<thead>", "").replace("</thead>", "")
    html = html.replace("<tbody>", "").replace("</tbody>", "")
    html = html.replace("（", "(").replace("）", ")")
    html = html.replace("<br>", "").replace("<br />", "").replace("<br/>", "").replace("</br>", "")
    html = html.replace("<th>", "<td>").replace("</th>", "</td>")
    return html

def extract_smiles_from_response(resp):
    pattern = re.compile(r'<smiles>(.*?)</smiles>', re.DOTALL)
    match = pattern.search(resp)
    return match.group(1).strip() if match else None

class ResultEvaluator:
    def __init__(self, res_path):
        self.success_set = None
        self._get_success_set(res_path)

    def _get_success_set(self, res_path):
        self.success_set = set()
        with open(res_path, 'r', encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                self.success_set.add(data["index"])

    def is_already_eval(self, index):
        return index in self.success_set

def extract_json(text):
    json_pattern = re.compile(r'```json\s*(.*?)\s*```', re.DOTALL)
    match = json_pattern.search(text)
    
    if match:
        s = match.group(1).strip()
        res = json.loads(s)
        return res
    else:
        s = ""
        start_index = text.find('{')
        end_index = text.rfind('}')
        if start_index != -1 and end_index != -1:
            s = text[start_index:end_index + 1]
            s = s.strip()
            res = json.loads(s)
            return res
        else:
            raise Exception("Parse error! Not find json in LLM response!")

def parse_html_table(html):
    soup = BeautifulSoup(html, 'html.parser')
    table = soup.find('table')

    max_row = 0
    max_col = 0
    rowspans = []

    for tr in table.find_all('tr'):
        current_col = 0
        cells = tr.find_all(['td', 'th'])
        for r in rowspans:
            if r > 0:
                current_col += 1
        for cell in cells:
            colspan = int(cell.get('colspan', 1))
            rowspan = int(cell.get('rowspan', 1))

            if rowspan > 1:
                rowspans += [rowspan - 1] * colspan
            current_col += colspan

        max_col = max(max_col, current_col)
        rowspans = [r - 1 if r > 0 else 0 for r in rowspans]
        max_row += 1

    table_data = [['' for _ in range(max_col)] for _ in range(max_row)]
    rowspans = [0] * max_col

    row_index = 0
    for tr in table.find_all('tr'):
        col_index = 0

        while col_index < max_col and rowspans[col_index] > 0:
            col_index += 1

        for cell in tr.find_all(['td', 'th']):
            cell_value = cell.get_text()
            colspan = int(cell.get('colspan', 1))
            rowspan = int(cell.get('rowspan', 1))

            for i in range(rowspan):
                for j in range(colspan):
                    while table_data[row_index + i][col_index + j] != '':
                        j += 1
                    table_data[row_index + i][col_index + j] = cell_value

            for i in range(colspan):
                if rowspan > 1:
                    rowspans[col_index + i] = rowspan - 1

            col_index += colspan

        rowspans = [r - 1 if r > 0 else 0 for r in rowspans]
        row_index += 1

    return table_data

def str_list2str(str_list):
    return ''.join(str_list)

def normalized_levenshtein_distance(str1, str2):
    if str1 == str2:
        return 1.0
    m, n = len(str1), len(str2)
    if m == 0:
        return 0.0 if n > 0 else 1.0
    if n == 0:
        return 0.0 if m > 0 else 1.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(
                    dp[i-1][j] + 1,
                    dp[i][j-1] + 1,
                    dp[i-1][j-1] + 1
                )
    edit_distance = dp[m][n]
    normalized_similarity = 1.0 - (edit_distance / max(m, n))
    return normalized_similarity

def evaluate_answer(question, ground_truth, model_answer):
    prompt = qa_answer_eval.replace("{Question}", question).replace("{Answer}", ground_truth).replace("{Model_Answer}", model_answer)
    
    try:
        eval_response = call_LLM([{"role": "user", "content": prompt}], model_name="gpt-4.1-nano-2025-04-14")
        
        try:
            eval_result = extract_json(eval_response)
            return eval_result.get("is_correct", "unknown")
        except json.JSONDecodeError:
            print(f"Failed to parse evaluation result: {eval_response}")
            return "unknown"
    except Exception as e:
        print(f"Error evaluating answer: {e}")
        return "unknown"

def is_valid_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False
