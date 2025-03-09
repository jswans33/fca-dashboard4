# pipeline_model

## Model Details

- **Version:** 1.0.0
- **Type:** ensemble
- **Created:** 2025-03-09T13:26:12.384841
- **Description:** A machine learning model for equipment classification.
- **Author:** NexusML Team

## Performance Metrics

- **accuracy:** 0.65
- **precision:** 0.7142857142857143
- **recall:** 0.5
- **f1:** 0.5882352941176471

## Model Parameters

- **bootstrap:** True
- **ccp_alpha:** 0.0
- **class_weight:** None
- **criterion:** gini
- **max_depth:** None
- **max_features:** sqrt
- **max_leaf_nodes:** None
- **max_samples:** None
- **min_impurity_decrease:** 0.0
- **min_samples_leaf:** 1
- **min_samples_split:** 2
- **min_weight_fraction_leaf:** 0.0
- **monotonic_cst:** None
- **n_estimators:** 100
- **n_jobs:** None
- **oob_score:** False
- **random_state:** 42
- **verbose:** 0
- **warm_start:** False

## Limitations

- This model may not perform well on data that is significantly different from the training data.
- The model's performance may degrade over time as data distributions change.

## Intended Use

This model is designed for classifying equipment based on descriptions and other features.
