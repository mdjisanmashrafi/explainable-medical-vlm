# MedXplain

### Prototype-Grounded Visual Evidence for Medical Vision-Language Models

MedXplain is an independent research exploration that investigates how **learned visual representations, data-driven prototypes, and similar-case retrieval** can provide additional visual context around medical Vision-Language Model (VLM) outputs.

The project explores a simple question:

> **How can learned representations and prototypes become visual evidence that a researcher can actually inspect?**

MedXplain currently uses **MedGemma** to obtain observations from medical images, extracts visual representations, organizes them into prototype groups, retrieves visually similar cases, and presents this context alongside the VLM output.

> **Important:** Prototype association and retrieved visual context do not establish that the VLM actually used those representations internally. MedXplain should therefore be viewed as an **evidence-exploration and research prototype**, not a validated clinical or faithful interpretability system.

---

## Research Motivation

Medical VLMs can generate useful observations from medical images, but their outputs can still be difficult to inspect from a visual perspective.

MedXplain explores whether a researcher can obtain additional context by connecting:

**Medical Image → VLM Observation → Visual Representation → Prototype → Similar Cases → Inspectable Evidence**

The goal is not simply to explain the model automatically, but to provide additional visual information that can be examined alongside its output.

---

## Key Components

### 1. Medical VLM

MedXplain uses **MedGemma** to process a medical image and generate an observation.

### 2. Visual Representation

The visual representation produced by the model is extracted into an embedding space.

### 3. Prototype Organization

The embedding space is organized using **K-Means clustering**, producing data-driven visual prototypes.

A prototype represents a recurring pattern within the learned representation space.

### 4. Similar-Case Retrieval

For a given input image, MedXplain retrieves visually similar reference cases from the embedding space.

### 5. Prototype-Grounded Evidence

Retrieved cases and their prototype association are presented alongside the VLM observation, allowing the researcher to inspect related visual patterns.

### 6. Grad-CAM Visualization

MedXplain also provides a **finding-guided Grad-CAM visualization** for additional visual inspection.

This visualization is exploratory and should not be interpreted as validated clinical localization.

---

## System Overview

```text
                    Medical Image
                         │
                         ▼
                     MedGemma
                         │
                         ▼
                VLM Observation
                         │
                         ▼
              Visual Representation
                         │
                         ▼
                 Embedding Space
                         │
                         ▼
                K-Means Prototypes
                         │
                ┌────────┴────────┐
                ▼                 ▼
        Similar-Case         Prototype
          Retrieval            Affinity
                │                 │
                └────────┬────────┘
                         ▼
              Inspectable Evidence
                         │
                         ▼
              Researcher Interface
```

---

## Research Questions

MedXplain currently investigates:

- Can learned visual representations provide useful context around VLM observations?
- Can data-driven prototypes organize visually related medical cases?
- Can similar-case retrieval make model outputs easier for researchers to inspect?
- How does prototype-grounded context relate to evidence attribution?
- Can prototype-based evidence eventually support more faithful multimodal reasoning?

---

## What I Built

The project includes the implementation of:

- Medical image preprocessing
- MedGemma-based visual representation extraction
- Visual embedding generation
- K-Means prototype organization
- Similarity-based image retrieval
- Prototype affinity analysis
- Grad-CAM finding-guided visualization
- Prototype-grounded evidence interface
- Interactive image questioning and evidence inspection

The system was developed as a research prototype to connect representation learning with interpretable visual evidence.

---

## Example Workflow

A medical image is first provided to the system.

```text
Input Image
     ↓
MedGemma
     ↓
Model Observation
```

The visual representation is then extracted:

```text
Visual Representation
     ↓
Embedding Space
     ↓
Prototype Assignment
```

The system retrieves visually similar reference cases:

```text
Input
  ↓
Similarity Search
  ↓
Top-K Similar Cases
  ↓
Prototype Context
```

Finally, the retrieved evidence is presented alongside the original VLM observation.

---

## Current Limitations

The current prototype has several limitations.

### Fixed Reference Collection

The retrieval and prototype index are currently constructed from a prepared reference collection. Arbitrary unseen images are therefore not yet supported as a fully online retrieval pipeline.

### Faithfulness

Prototype association does not prove that the VLM internally relied on that prototype or retrieved case.

### Retrieval Coverage

The quality of retrieved evidence depends on the coverage and diversity of the reference collection.

### Clinical Validation

The system is a research prototype and is not intended for clinical diagnosis or decision-making.

---

## Future Direction

The next stage of MedXplain is to move from visual context toward stronger evidence attribution.

Planned directions include:

- Online embedding and retrieval for unseen images
- Adaptive prototype formation
- Stronger evidence attribution
- Prototype-model alignment analysis
- Multimodal evidence grounding
- Adaptive multimodal reasoning
- Evaluation of whether retrieved evidence corresponds to model decisions
- Investigation of more faithful and trustworthy VLM reasoning

The broader research direction is:

**Prototype-grounded visual evidence → evidence attribution → adaptive multimodal reasoning → more trustworthy AI**

---

## Research Context

MedXplain is an independent research exploration motivated by closely related questions in:

- Interpretable AI
- Multimodal learning
- Visual representation learning
- Semantic prototypes
- Medical vision-language models
- Evidence-grounded reasoning
- Trustworthy AI

The project is particularly motivated by research questions surrounding how learned representations and semantic prototypes can support more interpretable and reliable AI systems.

---

## Dataset

The prototype was developed using the **VQA-RAD** medical Visual Question Answering dataset.

VQA-RAD contains radiology images paired with natural-language questions and answers and is commonly used for research on medical vision-language systems.

**Dataset:** [VQA-RAD](https://osf.io/89kps/)

For research use, please follow the dataset's original licensing and citation requirements.

---

## Technologies

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- MedGemma
- Scikit-learn
- K-Means
- Grad-CAM
- NumPy
- PIL
- Streamlit

---

## Project Structure

```text
MedXplain/
│
├── app/
│   └── app.py
│
├── data/
│   └── README.md
│
├── embeddings/
│   └── visual_embeddings.npy
│
├── prototypes/
│   ├── prototype_labels.npy
│   └── prototype_centroids.npy
│
├── retrieval/
│   └── retrieval.py
│
├── visualization/
│   └── gradcam.py
│
├── notebooks/
│   └── MedXplain.ipynb
│
├── requirements.txt
└── README.md
```

> Adjust the structure above to match the actual repository before publishing.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/MedXplain.git
cd MedXplain
```

Create an environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Demo

If the interface is implemented with Streamlit:

```bash
streamlit run app/app.py
```

Then open the local Streamlit URL shown in the terminal.

---

## Reproducibility

The project separates the representation, prototype, retrieval, and visualization stages so that each component can be inspected independently.

For reproducibility, the following artifacts should be generated or provided:

```text
Visual embeddings
       ↓
Prototype assignments
       ↓
Prototype centroids
       ↓
Similarity index
       ↓
Retrieved reference cases
       ↓
Evidence visualization
```

---

## Disclaimer

MedXplain is an academic research prototype.

It is not a medical diagnostic system and should not be used for clinical decision-making.

The visual evidence generated by the system is intended for research and inspection purposes. Prototype association, similarity retrieval, and Grad-CAM visualization should not be interpreted as proof of faithful internal reasoning by the underlying VLM.

---

## Citation

If you use this project in academic work, please cite the repository and the underlying models and datasets used in the implementation.

```bibtex
@software{medxplain,
  title  = {MedXplain: Prototype-Grounded Visual Evidence for Medical Vision-Language Models},
  author = {Md. Jisan Mashrafi},
  year   = {2026},
  url    = {https://github.com/YOUR_USERNAME/MedXplain}
}
```

---

## Author

**Md. Jisan Mashrafi**

Graduate Research Assistant  
MPhil in Artificial Intelligence  
Universiti Teknologi Malaysia

Research interests:  
Multimodal AI · Computer Vision · Representation Learning · Interpretable AI · Trustworthy AI

Website:  
https://mdjisanmashrafi.github.io/

---

## Status

**Research Prototype | Ongoing**

MedXplain is an ongoing exploration. The current implementation focuses on prototype-grounded visual evidence and retrieval, while future work will investigate stronger evidence attribution, adaptive prototypes, and more faithful multimodal reasoning.
