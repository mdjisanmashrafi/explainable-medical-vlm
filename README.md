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
