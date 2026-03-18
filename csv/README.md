# TCGA Textual & Omics Data

This folder contains the textual and omics data used in the integrative analysis for Head and Neck Cancer (HNC) classification and risk prediction. The data is a core component of the multi-omics dataset, complementing histopathology images and enabling comprehensive biomedical modelling.

## Dataset Overview

- **Total Number of Samples:** 82 patients with matched clinical and omics data
- **Data Size:** ~75 MB
- **Data Types:** Clinical, exposure, follow-up, methylation, mutations, pathology detail, transcriptomics
- **Patient IDs:** See [`common_82_samples.txt`](csv\common_82_samples.txt) for the list of included patients

### File Structure

- `clinical.csv` – Patient demographics, diagnosis, and clinical characteristics
- `exposure.csv` – Environmental and lifestyle exposures (e.g., smoking, alcohol)
- `follow_up.csv` – Longitudinal follow-up data including survival and recurrence events
- `methylation.csv` – DNA methylation profiles for tumor and normal samples
- `mutations.csv` – Somatic mutation data detailing gene variants
- `pathology_detail.csv` – Detailed histopathology annotations (tumor grade, subtype)
- `transcriptomics.csv` – Gene expression data (RNA-Seq)
- `common_82_samples.txt` – Patient IDs for the 82 samples with complete data

## Purpose & Usage

These files are used for:
- Feature engineering and extraction for machine learning models
- Integrating clinical and omics data with histopathology image features
- Classifying cancer stages and predicting clinical risk factors
- Supporting end-to-end pipelines for cancer detection and medical data modelling
