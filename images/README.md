

# TCGA Histopathology Images

This folder contains whole slide histopathology images in SVS format, used as part of the integrative analysis for Head and Neck Cancer (HNC) classification and risk prediction. The images are a key component of the multi-omics dataset, which also includes genomics, epigenomics, transcriptomics, and clinical data.

## Dataset Overview

- **Total Number of Samples:** 82 patients with matched clinical and omics data
- **Total Number of Slides:** 172 SVS files from 172 different patients (some patients have multiple slides, but only 82 have full textual/omics data)
- **Data Size:** >20 GB of histopathology images
- **Format:** Aperio SVS (whole slide imaging)

## Purpose & Usage

These slides are used for:
- Extracting histopathological features for machine learning models
- Integrating image-derived features with genomics, epigenomics, and transcriptomics data
- Classifying cancer stages and predicting clinical risk factors
- Supporting end-to-end pipelines for cancer detection and medical data modelling

## Integration with the Repository

The images are processed alongside categorical and omics data (see main [README](../README.md)). Feature extraction, preprocessing, and validation are performed to enable robust classification and risk prediction. The integrative model leverages high-dimensional biomedical data to address real-world clinical challenges in HNC detection.

## Notes

- Only 82 patients have complete textual/omics data; slides for other patients are included for completeness but may not be used in modelling.
- For details on preprocessing, feature engineering, and model training, refer to the main [README](../README.md) and relevant notebooks in the root directory.
