# Model Comparison

The final test set was held out before model fitting. Cross-validation was run only on the training portion.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Confusion matrix |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 0.714 | 0.750 | 0.750 | 0.750 | 0.667 | `[[2, 1], [1, 3]]` |
| word_tfidf | 0.571 | 0.571 | 1.000 | 0.727 | 0.750 | `[[0, 3], [0, 4]]` |
| char_tfidf | 0.571 | 0.571 | 1.000 | 0.727 | 1.000 | `[[0, 3], [0, 4]]` |
| combined | 0.857 | 0.800 | 1.000 | 0.889 | 0.667 | `[[2, 1], [0, 4]]` |

## Cross-validation

### baseline
- accuracy: 0.719 +/- 0.124
- precision: 0.732 +/- 0.191
- recall: 0.867 +/- 0.094
- f1: 0.775 +/- 0.091
- roc_auc: 0.740 +/- 0.205

### word_tfidf
- accuracy: 0.537 +/- 0.026
- precision: 0.537 +/- 0.026
- recall: 1.000 +/- 0.000
- f1: 0.698 +/- 0.022
- roc_auc: 0.907 +/- 0.082

### char_tfidf
- accuracy: 0.537 +/- 0.026
- precision: 0.537 +/- 0.026
- recall: 1.000 +/- 0.000
- f1: 0.698 +/- 0.022
- roc_auc: 0.883 +/- 0.085

### combined
- accuracy: 0.789 +/- 0.150
- precision: 0.750 +/- 0.177
- recall: 1.000 +/- 0.000
- f1: 0.846 +/- 0.109
- roc_auc: 0.740 +/- 0.205
