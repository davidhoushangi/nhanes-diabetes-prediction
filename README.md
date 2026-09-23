#  predicting a reported diabetes diagnosis

This Python project learns from the **2017–2018 US National Health and Nutrition Examination Survey (NHANES)**. It predicts whether an adult says a clinician has told them they have diabetes, using age, recorded sex, BMI, and waist circumference. It is an educational analysis, **not** a diabetes diagnosis or a future-risk forecast.

## Repo Structure
```
nhanes-diabetes-prediction/
├── README.md
├── train.py
├── requirements.txt
├── .gitignore
└── results/
    ├── classification_report.txt
    ├── evaluation.png
    └── metrics.json
```

## Run it

1. Install Python 3.10 or newer.
2. In this folder, run `python -m pip install -r requirements.txt`.
3. Run `python train.py`.

The first run downloads three original CDC `.XPT` files into `data/`. Outputs appear in `results/`. Internet access is needed only for the first download.

## How it works

- Join demographics (`DEMO_J`), body measures (`BMX_J`), and diabetes questionnaire (`DIQ_J`) on the anonymous survey participant ID `SEQN`.
- Keep adults aged 20 or older. `DIQ010=1` is the positive class and `DIQ010=2` is the negative class. Exclude borderline, refused, and unknown responses.
- Split participants into stratified training (75%) and held-out testing (25%) with random seed 42.
- Impute missing measurements and standardize numerical fields *inside* the training pipeline. Train logistic regression and compare average precision to a prevalence-only baseline.
- Save sample counts, held-out ROC AUC, average precision, Brier score, a confusion matrix, a text classification report, and a figure.

## What to look at

- `results/metrics.json`: numbers from the held-out test participants.
- `results/evaluation.png`: ROC curve and confusion matrix.
- `results/classification_report.txt`: precision and recall for each group.

**Accuracy alone is misleading:** most respondents did not report diabetes. Read the positive-class recall and precision together. The 0.5 classification threshold is an illustration, not a clinical cutoff.

## Important limitations

The outcome is a *self-reported previous diagnosis*, not blood-test-confirmed diabetes; people with undiagnosed diabetes may be in the negative group. Measurements and the report come from the same survey, so the model cannot predict future disease. NHANES uses a complex survey design: this introductory exercise does **not** apply survey weights or estimate population-wide prevalence or performance. A random split within one survey cycle does not demonstrate performance in another year, hospital, or population. Model probabilities should not be interpreted as clinically validated personal risk.

The source files are public, but check the CDC documentation before publishing derived data. Keep the downloaded `data/` files out of your GitHub repository; anyone can reproduce the download using this script.

## Primary Data sources

- [NHANES datasets and documentation](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/)
- [Diabetes questionnaire, DIQ_J](https://wwwn.cdc.gov/nchs/data/nhanes/public/2017/datafiles/DIQ_J.htm)
- [Demographics, DEMO_J](https://wwwn.cdc.gov/nchs/data/nhanes/public/2017/datafiles/DEMO_J.htm)
- [Body measures, BMX_J](https://wwwn.cdc.gov/nchs/data/nhanes/public/2017/datafiles/BMX_J.htm)
