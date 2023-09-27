# Assignment1-BMI550
Assignment 1: BMI 550— Applied BioNLP
Author: Chase Fensore

## Python Packages
- fuzzywuzzy==0.18.0
- nltk==3.8.1
- pandas==2.0.0
- python_Levenshtein==0.22.0
- scikit_learn==1.3.1
- thefuzz==0.20.0

## Hyperparameters for Experiments
1. **Preprocessing**: as described in the report, 3 combinations are used. 
- P0: lowercasing and word tokenization. TO USE: pass "none" to preprocessing argument of predict_test_CUIs (line 423).
- P1: P0 + punctuation removal. TO USE: pass "pp1" to preprocessing argument of predict_test_CUIs (line 423).
- P2: P1 + Porter stemming. TO USE: pass "pp2" to preprocessing argument of predict_test_CUIs (line 423).
3. **Similarity metrics**:
- Levenshtein distance: to use, pass "Levenshtein" to sim_metric argument of predict_test_CUIs (line 423).
- Token sort ratio: to use, pass "token_sort_ratio" to sim_metric argument of predict_test_CUIs (line 423).
3. **Threshold** (for similarity metric):
- Token sort ratio: to adjust between 0-100, change _min_pred_thresh_ argument of predict_test_CUIs (line 423).
- Levenshtein distance: to adjust between 0-1.0, change _min_pred_thresh_ argument of predict_test_CUIs (line 423).


## To Run
1. ```conda install requirements.txt```
2. To perform automatic symptom detection on Assignment1GoldStandardSet.xlsx:
- Run: ```python rulebased_system_Fensore.py```. (Notes: line 409: must set file to read Assignment1GoldStandardSet.xlsx, line 423: set desired hyperparameters described above, line 424: change output file name, if desired.)
- New output will be stored in: data/result-Assignment1GoldStandardSet.xlsx.
- My best-performing output is currently stored in: results/result-Assignment1GoldStandardSet.xlsx.
4. To perform automatic symptom detection on UnlabeledSet.xlsx:
- Run: ```python rulebased_system_Fensore.py```. (Notes: line 409: must set file to read UnlabeledSet.xlsx, line 423: set desired hyperparameters described above, line 424: change output file name, if desired.)
- New output will be stored in data/result-UnlabeledSet.xlsx
- Existing output is stored in: results/result-UnlabeledSet.xlsx.
