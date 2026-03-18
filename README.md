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

## Contributors

- Data Engineer: [Ibrahim Hussain](https://github.com/ib-hussain)
- ML Engineer: [Mahd Kazmi](https://github.com/Mahdkazmi)

---
For questions or further information, see the documentation or contact the project maintainers.
