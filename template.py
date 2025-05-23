get_smiles = """
## Instruction
Identify the molecules in this image and return them to me in smiles format. 

## Format
The answer is wrapped in <smiles></smiles>.

"""

tsr_html_prompt = """
Identify the structure of the table and return it to me in HTML format.
Note: 
1. Use the <thead> and <tbody> tags to distinguish the table header from the table body.
2. Use only five tags: <table>, <thead>, <tr>, <td>, and <tbody>.
3. If a molecule graph appears in a cell, use a SMILES expression instead, and use [#smiles#] to wrap the smile molecular formula. For example, [#smiles#]C1=CC=CC=C1[#smiles#] represents benzene.
"""

title_prompt = """
## Instruction
Extract the title from this chemical document image.


## Format
Return the results in the following JSON format:
```json
{
    "chain_of_thought": "your chain of thought about how you get the final result.",
    "title": "Title text"
}
```


## Answer
```json
"""

anno_prompt = """
## Instruction
Extract the annotations from this chemical document image.


## Format:
Return the results in the following JSON format:
```json
{
    "chain_of_thought": "your chain of thought about how you get the final result.",
    "annotations": "Annotation text"
}
```


## Answer
```json
"""


generate_statistic_question = """
Based on the table in the picture and the content of the table, generate 5 statistical questions (compare, find the maximum value, calculate the sum and mean of values). Please give the question-answer pair. Return it to me in json format.
category:
1. compare
2. max
3. sum
4. mean


## Table
{Table_html}


## Format:
```json
{
    "chain_of_thought": "your chain of thought about how you get the final result.",
    "QA": [{"question": "question 1","answer": "answer 1","category": "..."},{"question": "question 2","answer": "answer 2","category": "..."}, ... ]
}
```

## Answer
```json
"""

get_cell_value_by_position = """
### instruction
Please identify the table in the picture, and retrieve the corresponding cell according to the row and column coordinates given and output.
Note: 
1. The coordinates of the cell are represented by the row and column numbers of the cell. The row and column numbers start from 1.
2. The cell value is a string, and the output format is a string.
3. If the cell is empty, please return an empty string.
4. All operations are performed on the entire table, including the head and body. When counting the number of rows and columns, the header and body of the table are also counted. If there are headers and columns, The row count starts from the header row. The columns count starts from the header columns.


Please return the answer in the following JSON format:
```json
{"chain_of_thought": "Your thought process for complete this task.","content": "The cell content."}
```


## Answer
```json"""


find_cell_position_by_value = """
### instruction
Please identify the table in the picture, and retrieve the corresponding cell position according to the cell value given and output.
Note:
1. The cell position is represented by the row and column numbers of the cell. The row and column numbers start from 1.
2. If the cell is not found, please return an empty string.
3. All operations are performed on the entire table, including the head and body. When counting the number of rows and columns, the header and body of the table are also counted. If there are headers and columns, The row count starts from the header row. The columns count starts from the header columns.


Please return the answer in the following JSON format:
```json
{"row_index": 0,"col_index": 0}
```

## Answer
```json"""

detect_table_dimensions = """
### instruction
Please identify the table in the picture, and retrieve the dimensions of the table and output.
Note:
1. The dimensions of the table are represented by the number of rows and columns.
2. The dimensions of the table are two integers. The row and column numbers start from 1.
3. All operations are performed on the entire table, including the head and body. When counting the number of rows and columns, the header and body of the table are also counted. If there are headers and columns, The row count starts from the header row. The columns count starts from the header columns.
4. Your answer must be returned in the following json format. Your answers must be placed on one line, without line breaks.


Please return the answer in the following JSON format:
```json
{"rows": 0,"columns": 0}
```

## Answer
```json"""


qa_prompt_base_html = """
## instruction
Please answer the question based on the html content of the table.


## Table
{Table_html}


## Format:
```json
{
    "chain_of_thought": "your chain of thought about how you get the final result.",
    "answer": "answer"
}
```


## Question
{Question}


## Answer:
```json
"""


qa_prompt_base_image = """
## instruction
Please answer the question based on the image of the table.


## Format:
```json
{
    "chain_of_thought": "your chain of thought about how you get the final result.",
    "answer": "answer"
}
```


## Question
{Question}


## Answer:
```json
"""

qa_prompt_base_hybrid = """
## instruction
Please answer the question based on the html content of the table and the image of the table.


## Table
{Table_html}


## Format:
```json
{
    "chain_of_thought": "your chain of thought about how you get the final result.",
    "answer": "answer"
}
```


## Question
{Question}


## Answer:
```json
"""

qa_answer_eval = """
## instruction
Please evaluate the answer based on the question and the answer. If the answer is correct, please return "correct". If the answer is incorrect, please return "incorrect".
If the answer is unable to answer the question, please make sure the model's answer is refused to answer the question.

## Question
{Question}


## Answer
{Answer}


## Model's Answer
{Model_Answer}


## Format:
```json
{
    "chain_of_thought": "your chain of thought about how you get the final result.",
    "is_correct": "correct or incorrect"
}
```


## Answer
```json
"""


generate_personalization_question = """
Based on the table in the picture and the summary of the table, generate 3 personalization questions. Please give the question-answer pair. Return it to me in json format.
Note:
1. The question should be related to the specific meaning of the table.
2. The answer should be a specific value, not a range or a description.
3. Do not generate basic statistical questions like compare, max, sum, or mean.
4. Choose the more difficult questions as your response while keeping your answer closely related to the content of the table. Be sure to think over and over again to make sure that the answers to the difficult questions are correct.

## Table
{Table_html}


## Table Summary
{Table_summary}


## Format:
```json
{
    "chain_of_thought": "your chain of thought about how you get the final result.",
    "QA": [{"question": "question 1","answer": "answer 1"},{"question": "question 2","answer": "answer 2"}, ... ]
}
```

## Answer
```json
"""

generate_table_meaning = """
Based on the table in the picture and the content of the table, please generate a concise summary that explains the main purpose and content of the table. The summary should include what the table is about, what research or data it presents, and its key findings or implications.

## Table
{Table_html}


## Format:
```json
{
    "chain_of_thought": "your chain of thought about how you get the final result.",
    "summary": "summary"
}
```

## Answer
```json
"""

unable_to_answer_reason_prompt = """
## Instruction
You are an expert in the field of question answering. Now I have a question based on an image, which is marked as unanswerable, that is, the question given cannot be answered based on the image. I need you to analyze the reasons why it cannot be answered. For example: non-existent rows, no columns of that color, non-existent substances, etc.

## Question
{Question}


## Format:
```json
{
    "chain_of_thought": "your chain of thought about how you get the final result.",
    "reason": "reason"
}
```


## Answer
```json
"""

unable_to_answer_reason_categories = """
## Instruction
You are an expert in the field of question answering. Now I have an image-based question that is marked as unanswerable, that is, the question given cannot be answered based on the image. I already have the categories why it cannot be answered, and I need you to analyze the type it belongs to.



## Question
{Question}

## Reason
{Reason}

## Categories List
["Missing Col/Row", "Missing Style(color or bold)", "Ambiguity", "Misplacement"]

## Format:
```json
{
    "chain_of_thought": "your chain of thought about how you get the final result.",
    "Categories": "",
}
```


## Answer
```json
"""