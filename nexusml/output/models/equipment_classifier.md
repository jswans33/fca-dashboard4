# Equipment Classifier

## Model Details

- **Version:** 1.0.0
- **Type:** Classification
- **Description:** A random forest model for classifying equipment based on descriptions.
- **Authors:** NexusML Team
- **License:** Proprietary
- **Created:** 2025-03-08T23:45:12.733050

## Intended Use

This model is designed for classifying equipment based on descriptions and other features.

## Training Data

The model was trained on a dataset of equipment descriptions and associated metadata.

## Evaluation Results

### Overall Metrics

- **accuracy_mean:** 0.6
- **precision_mean:** 0.6
- **recall_mean:** 0.6
- **f1_score_mean:** 0.6

### Per-Column Metrics

#### category_name

- **accuracy:** 0.0
- **precision:** 0.0
- **recall:** 0.0
- **f1_score:** 0.0

#### uniformat_code

- **accuracy:** 1.0
- **precision:** 1.0
- **recall:** 1.0
- **f1_score:** 1.0

#### mcaa_system_category

- **accuracy:** 1.0
- **precision:** 1.0
- **recall:** 1.0
- **f1_score:** 1.0

#### System_Type_ID

- **accuracy:** 1.0
- **precision:** 1.0
- **recall:** 1.0
- **f1_score:** 1.0

#### Equip_Name_ID

- **accuracy:** 0.0
- **precision:** 0.0
- **recall:** 0.0
- **f1_score:** 0.0


## Model Parameters

- **memory:** None
- **steps:** [('preprocessor', ColumnTransformer(transformers=[('text', TfidfVectorizer(max_features=1000),
                                 'combined_text'),
                                ('numeric', StandardScaler(),
                                 ['service_life', 'CategoryID', 'OmniClassID',
                                  'UniFormatID', 'MasterFormatID']),
                                ('categorical',
                                 OneHotEncoder(handle_unknown='ignore'),
                                 ['equipment_tag', 'manufacturer', 'model'])])), ('classifier', MultiOutputClassifier(estimator=RandomForestClassifier(max_depth=10,
                                                       random_state=42)))]
- **transform_input:** None
- **verbose:** False
- **preprocessor:** ColumnTransformer(transformers=[('text', TfidfVectorizer(max_features=1000),
                                 'combined_text'),
                                ('numeric', StandardScaler(),
                                 ['service_life', 'CategoryID', 'OmniClassID',
                                  'UniFormatID', 'MasterFormatID']),
                                ('categorical',
                                 OneHotEncoder(handle_unknown='ignore'),
                                 ['equipment_tag', 'manufacturer', 'model'])])
- **classifier:** MultiOutputClassifier(estimator=RandomForestClassifier(max_depth=10,
                                                       random_state=42))
- **preprocessor__force_int_remainder_cols:** True
- **preprocessor__n_jobs:** None
- **preprocessor__remainder:** drop
- **preprocessor__sparse_threshold:** 0.3
- **preprocessor__transformer_weights:** None
- **preprocessor__transformers:** [('text', TfidfVectorizer(max_features=1000), 'combined_text'), ('numeric', StandardScaler(), ['service_life', 'CategoryID', 'OmniClassID', 'UniFormatID', 'MasterFormatID']), ('categorical', OneHotEncoder(handle_unknown='ignore'), ['equipment_tag', 'manufacturer', 'model'])]
- **preprocessor__verbose:** False
- **preprocessor__verbose_feature_names_out:** True
- **preprocessor__text:** TfidfVectorizer(max_features=1000)
- **preprocessor__numeric:** StandardScaler()
- **preprocessor__categorical:** OneHotEncoder(handle_unknown='ignore')
- **preprocessor__text__analyzer:** word
- **preprocessor__text__binary:** False
- **preprocessor__text__decode_error:** strict
- **preprocessor__text__dtype:** <class 'numpy.float64'>
- **preprocessor__text__encoding:** utf-8
- **preprocessor__text__input:** content
- **preprocessor__text__lowercase:** True
- **preprocessor__text__max_df:** 1.0
- **preprocessor__text__max_features:** 1000
- **preprocessor__text__min_df:** 1
- **preprocessor__text__ngram_range:** (1, 1)
- **preprocessor__text__norm:** l2
- **preprocessor__text__preprocessor:** None
- **preprocessor__text__smooth_idf:** True
- **preprocessor__text__stop_words:** None
- **preprocessor__text__strip_accents:** None
- **preprocessor__text__sublinear_tf:** False
- **preprocessor__text__token_pattern:** (?u)\b\w\w+\b
- **preprocessor__text__tokenizer:** None
- **preprocessor__text__use_idf:** True
- **preprocessor__text__vocabulary:** None
- **preprocessor__numeric__copy:** True
- **preprocessor__numeric__with_mean:** True
- **preprocessor__numeric__with_std:** True
- **preprocessor__categorical__categories:** auto
- **preprocessor__categorical__drop:** None
- **preprocessor__categorical__dtype:** <class 'numpy.float64'>
- **preprocessor__categorical__feature_name_combiner:** concat
- **preprocessor__categorical__handle_unknown:** ignore
- **preprocessor__categorical__max_categories:** None
- **preprocessor__categorical__min_frequency:** None
- **preprocessor__categorical__sparse_output:** True
- **classifier__estimator__bootstrap:** True
- **classifier__estimator__ccp_alpha:** 0.0
- **classifier__estimator__class_weight:** None
- **classifier__estimator__criterion:** gini
- **classifier__estimator__max_depth:** 10
- **classifier__estimator__max_features:** sqrt
- **classifier__estimator__max_leaf_nodes:** None
- **classifier__estimator__max_samples:** None
- **classifier__estimator__min_impurity_decrease:** 0.0
- **classifier__estimator__min_samples_leaf:** 1
- **classifier__estimator__min_samples_split:** 2
- **classifier__estimator__min_weight_fraction_leaf:** 0.0
- **classifier__estimator__monotonic_cst:** None
- **classifier__estimator__n_estimators:** 100
- **classifier__estimator__n_jobs:** None
- **classifier__estimator__oob_score:** False
- **classifier__estimator__random_state:** 42
- **classifier__estimator__verbose:** 0
- **classifier__estimator__warm_start:** False
- **classifier__estimator:** RandomForestClassifier(max_depth=10, random_state=42)
- **classifier__n_jobs:** None

## Limitations

This model may not perform well on data that is significantly different from the training data.

## Ethical Considerations

This model should be used responsibly and in accordance with applicable laws and regulations.
