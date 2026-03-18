<h1 align="center">TCGA Integrative Analysis</h1>

<p align="center">
	<img src="images\histopathology.png" alt="Whole Slide Histopathology Image" >
</p>

## Global Cancer Context

Approximately 9.7 million people die from cancer each year worldwide (2022), representing nearly 1 in 6 deaths globally. Among men, Head and Neck Cancer (HNC) ranks as the fifth most common cancer overall, while in women it ranks 12th. 


## Project Aim & Overview

Developed an integrative machine learning model to classify Head and Neck Cancer (HNC) stages and predict risk factors using multi-omics data (genomics, epigenomics, transcriptomics) and histopathology images. Built end-to-end in under 24 hours, combining high-dimensional biomedical data with rigorous feature engineering to address real-world clinical challenges.

### Classification & Prediction Goals
- Classify cancer vs non-cancer
- Classify stages of cancer
- Predict risk factors (smoking, alcohol, etc.)

## Omics & Imaging Data Explained

- **Images:** H&E stained images show tissue morphology ([.svs files])
- **Genome:** Mutation profiles indicate which genes are altered and their frequency ([.maf files])
- **Transcriptome:** RNA transcript counts reveal gene activity levels in each sample ([.tsv files])
- **Epigenome:** DNA methylation scores (0 to 1) reflect chemical tagging that can silence genes ([.tsv files])


## Project Overview


**Goal:** Develop integrative machine learning models for HNC stage classification and risk prediction
**Data Types:**
	- Genomics, epigenomics, transcriptomics (textual/omics data)
	- Whole slide histopathology images (SVS format)
	- Clinical and exposure data
**Pipeline:** Preprocessing, feature extraction, integration, model training, and validation
**Scale:**
	- \>20 GB histopathology images (172 slides, 82 patients with full data)
	- ~75 MB textual/omics data (82 patients)


## Repository Structure

- [images](images/README.md): Whole slide histopathology images in SVS format, used for extracting image-derived features
- [csv](csv/README.md): Textual and omics data (clinical, exposure, follow-up, methylation, mutations, pathology detail, transcriptomics)
- Notebooks: Data cleaning, feature engineering, and model training pipelines
- Other files: Combined feature matrices, cohort information, and supporting documentation

## Dataset Details

- **Patients:** 82 with complete multi-omics and clinical data
- **Slides:** 172 SVS files (some patients have multiple slides; only 82 have full textual/omics data)
- **Patient IDs:** See `csv/common_82_samples.txt`

### Omics & Clinical Data
- `clinical.csv`: Demographics, diagnosis, clinical characteristics
- `exposure.csv`: Environmental/lifestyle exposures
- `follow_up.csv`: Longitudinal follow-up (survival, recurrence)
- `methylation.csv`: DNA methylation profiles
- `mutations.csv`: Somatic mutation data
- `pathology_detail.csv`: Histopathology annotations
- `transcriptomics.csv`: Gene expression (RNA-Seq)

### Histopathology Images
- Aperio SVS format
- Used for extracting histopathological features

## Purpose & Usage


Processed and analysed (>20GB) histopathology image dataset alongside (>150MB) categorical patient data
Trained classification models to detect cancer stages and clinical patterns
Applied preprocessing, feature extraction, and validation pipelines for high-dimensional medical data
Feature engineering from multi-omics and image data
Integration of clinical, omics, and histopathology features
Training and validation of classification models for cancer stage and risk prediction
End-to-end pipelines for robust biomedical modelling

## Data Engineering Methodology

The project's robust data pipelines systematically clean, align, and integrate high-dimensional clinical and multi-omics data. Key data operations detailed in the repository's data wrangling notebooks include:

### 1. Data Cleaning & Dimensionality Reduction
- **Missing Value Handling:** Systematically drop features with exceedingly high missing rates (e.g., >90% or >95% nulls), such as predominantly empty pathology or exposure descriptions. Completely null columns are identified and removed automatically.
- **Low Variance Filtering:** Drop variables containing singular unique values (non-informative features) while computationally preventing the exclusion of essential clinical keys.
- **Mutation Curation:** Filter for functional (non-synonymous) mutations (e.g., Missense, Nonsense, Frame Shifts) and calculate the Variant Allele Frequency (VAF) from depth counts to distill actionable genomic variants.

### 2. Clinical & Pathology Data Wrangling
- **Clinical Integration:** Perform full outer joins to seamlessly unify `clinical`, `exposure`, and `follow_up` datasets per patient without data loss.
- **Feature Engineering:** Engineer composite demographic and diagnostic indicators, such as combined `disease_index` (disease type and index date) and unified `ethnicity_race` arrays. Standardize case-level identifiers (12-char TCGA barcodes) across disparate clinical sources.
- **Curated Filtering:** Generate consolidated reports of dropped parameters and yield focused, smaller 'curated' clinical cohort dataframes for streamlined downstream ML integration.

### 3. Multi-Omics Signal Alignment (Transcriptomics & Methylation)
- **ID Standardization:** Standardize sample-level TCGA identifiers (15-char formats) uniformly across methylation and transcriptomics records.
- **Epigenetic Equation:** Filter for intersecting sequenced genes and patient samples across both datasets to generate an integrative *combined epigenetic score* modeled as `Combined Score = (1 - Methylation) * Transcriptomics`, representing gene silencing impact on expression.

### 4. Integrative Merging & Matrix Generation
- **Mutation-Epigenetic Interaction:** Transform raw curated mutations into binary, patient-by-gene mutation inclusion matrices. Multiply this against the corresponding combined epigenetic scores to create a fully quantified interaction matrix, factoring out normal tissue equivalents.
- **Unified Modeling Matrices:** Produce cohesive final datasets merging the functional clinical/pathology context, filtered gene mutation statuses, and computed epigenetic signals into unified, analysis-ready CSV structures.

### 5. Final Output Datasets
The culmination of these operations generates four primary analytical CSVs located in the repository's root, structurally optimized for machine learning ingestion:
- `Pathology_Cohorts_v6.csv`: Consolidates the comprehensively curated clinical background data linked with specific histopathological indicators.
- `combined_epigenetic_score_matrix.csv`: Quantifies transcriptomic expression continuously weighted and mathematically modulated by localized DNA methylation levels per gene.
- `mutations_plus_epigenetics.csv`: A comprehensive concatenated structure of patient binary mutation vectors concatenated dynamically with their aligned epigenetic scores.
- `mutation_epigenetic_interaction_matrix.csv`: The multiplicative interaction matrices extracting the integrated functional interplay specifically isolating active gene mutations against their accompanying epigenetic constraints.

## Contributors

- Data Engineer: [Ibrahim Hussain](https://github.com/ib-hussain)
- ML Engineer: [Mahd Kazmi](https://github.com/Mahdkazmi)

---
For questions or further information, see the documentation or contact the project maintainers.
