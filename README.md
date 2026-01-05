# Tracing Consultation Inputs in Legislative Drafting

An embedding-based case study analyzing stakeholder influence on the Geothermiebeschleunigungsgesetz (Geothermal Acceleration Act).

## Overview

This repository contains a computational text analysis of how stakeholder comments align with successive drafts of German legislation. Using NLP techniques including TF-IDF similarity and sentence embeddings, we trace the evolution of legal text through consultation stages to identify patterns of semantic convergence between stakeholder input and final legislative wording.

## Research Question

**To what extent do legislative drafts semantically converge with stakeholder comments throughout the revision process?**

## Methodology

### Data

The analysis combines four draft versions of the Geothermiebeschleunigungsgesetz with two stages of stakeholder consultation:

- **Draft v0** (03.07.2025): Initial ministerial draft (*Referentenentwurf*)
- **Draft v1** (15.08.2025): Internal government revision
- **Draft v2** (01.10.2025): Parliamentary introduction
- **Draft v3** (03.12.2025): Final committee report

**Stakeholder Input:**
- **Comments v1**: 33 organizations submitted structured, article-level comments via the BMWK online platform
- **Comments v2**: Unstructured letters from parliamentary expert hearing (November 2025)

### Analysis Approach

**1. Lexical Similarity (TF-IDF Cosine)**
- Measures word-level changes between consecutive draft versions
- Uses paragraph-level alignment with unigrams and bigrams
- Identifies where and when text was revised

**2. Semantic Similarity (Sentence Embeddings)**
- Generates embeddings using `paraphrase-multilingual-MiniLM-L12-v2`
- Computes cosine similarity between stakeholder comments and law provisions
- Tracks alignment shift: Δ*S* = cos(*c*, *d*<sub>v3</sub>) - cos(*c*, *d*<sub>v0</sub>)

## Key Findings

### Three-Phase Drafting Trajectory

1. **Initial Revision (v0→v1)**: Substantial rewording with mean similarity of 0.86; ~32% of paragraphs near-identical, ~2% fully replaced
2. **Stabilization (v1→v2)**: High lexical stability with mean similarity of 0.99; over 80% of paragraphs unchanged
3. **Targeted Refinement (v2→v3)**: Selective editing with mean similarity of 0.88; ~55% unchanged, ~8% fully replaced

### Semantic Alignment

- **Stable Baseline**: Most organizations show alignment shifts (Δ*S*) close to zero, indicating the consultation process did not fundamentally redirect the bill's semantic orientation
- **Positive Convergence**: A few organizations (e.g., Stadtwerke München, DVGW) show modest increases in similarity (+0.05 to +0.15), suggesting their language became more aligned with later drafts
- **Document-Level**: Stakeholder letters from the parliamentary hearing show consistent positive shifts (+0.02 to +0.05) toward the final draft

### Stakeholder Characteristics

- **Industry associations** (*Wirtschaftsverbände*) dominate participation, with BDEW and VKU submitting the most comments (21 and 18 respectively)
- **State ministries** (*Länderbehörden*) submit more frequent but concise comments, suggesting targeted modification requests rather than extensive argumentation
- Wide variation in similarity baselines (0.34 to 0.77) reflects heterogeneity in submission styles

## Repository Structure

```
nlp_legislative_consultations/
├── data/                     # Legislative drafts and consultation data
├── notebooks/                # Analysis notebooks
│   ├── 01_org_analysis.ipynb         # Organization-level statistics
│   ├── 02_versions_similarity.ipynb  # Lexical change analysis (TF-IDF)
│   └── 03_embedding_analysis.ipynb   # Semantic alignment (embeddings)
├── scripts/                  # Data acquisition and preprocessing
├── writeup/                  # LaTeX research paper
└── requirements.txt          # Python dependencies
```

## Analysis Notebooks

1. **Organization Analysis**: Descriptive statistics of stakeholder submissions (frequency, length, article coverage)
2. **Version Similarity**: TF-IDF cosine similarity tracking lexical changes across draft versions
3. **Embedding Analysis**: Semantic similarity using sentence transformers to measure alignment between comments and drafts

## Installation

```bash
pip install -r requirements.txt
```

Key dependencies: `pandas`, `numpy`, `scikit-learn`, `sentence-transformers`, `matplotlib`, `seaborn`

## Citation

This analysis is part of a research project at Hertie School. For detailed methodology and discussion, see `writeup/main.tex`.
