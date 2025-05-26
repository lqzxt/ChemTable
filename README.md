# 🧪 ChemTable: Benchmarking Multimodal LLMs on Recognition and Understanding over Chemical Tables


ChemTable is the first large-scale benchmark designed to test the capabilities of multimodal large language models (MLLMs) in understanding **real-world chemical tables**—one of the most information-dense and visually complex formats in scientific literature.

<img src="img/dataset_info.png" width="60%" />



> 📘 Built from over 1,300 tables from high-impact chemistry journals, ChemTable combines **visual, textual, symbolic**, and **domain-specific** information to push the boundaries of scientific AI.

---

## 🚀 Key Features

- **Multimodal Benchmark**  
  Combines symbolic chemical formulas, table structures, visual molecule diagrams, and scientific text.

- **Two Core Tasks**  
  1. **Table Recognition**: Detect structure, extract content, and identify molecules.  
  2. **Table Understanding**: Answer descriptive and reasoning-based questions from tables.

- **Challenging QA Dataset**  
  Includes 9,000+ questions (descriptive + reasoning), curated with a mix of human annotation and LLM-assisted synthesis.

---

## 🧩 Dataset Structure

![case](img/dataset_case.png)

- **Table Types**: Reaction optimization, substrate screening, property comparison, molecular structure tables, and more.
- **Visual Annotations**: Bounding boxes, styles (bold/color), molecule diagrams.
- **Logical Annotations**: Row/column positions, cell values, chemical metadata.

---

## 🏗️ Tasks

### 📐 Table Recognition
| Subtask               | Description                                        | Metric     |
|----------------------|----------------------------------------------------|------------|
| Value Retrieval       | Locate exact content at given (row, column)       | Accuracy   |
| Position Retrieval    | Infer position from given content                 | Accuracy   |
| Molecular Recognition | Identify SMILES from embedded diagrams            | Tanimoto   |



### 🤖 Table Understanding

<img src="img/qa_case.png" width="60%" />

---

## 🔬 Experimental Results

<img src="img/tr_res.png" width="60%" />
<img src="img/result.png" width="60%" />

<image>