# Titanic Survival Prediction Using R — Week 3

## Internship
Virtual R Data Analyst Internship — Yuva Learn Platform, Pune

## Project objective
Perform hypothesis testing and build a logistic-regression classification model to predict Titanic passenger survival.

## Dataset
The project uses the cleaned Titanic dataset from Week 1 (891 rows). The target is `Survived` (Yes/No). Predictors: `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, and `Embarked`.

## Methods
- Exploratory analysis and summary statistics
- Chi-square tests for gender/class versus survival
- Shapiro-Wilk normality checks and appropriate two-sample testing for age
- Wilcoxon rank-sum test for fare
- Spearman correlation analysis
- Stratified 80/20 train/test split
- 5-fold stratified cross-validation
- Logistic regression classifier
- Accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix and ROC curve

## Current reference results
These are reference results calculated on the supplied cleaned dataset with a stratified 80/20 split and fixed random seed. Re-run the R script locally to reproduce results in your installed R package versions.

| Metric | Reference value |
|---|---:|
| Accuracy | 0.8045 |\n| Precision | 0.7931 |\n| Recall | 0.6667 |\n| F1-score | 0.7244 |\n| ROC-AUC | 0.8432 |\n| Mean 5-fold CV ROC-AUC | 0.8528 |\n| SD 5-fold CV ROC-AUC | 0.0217 |\n
## Folder structure
- `Dataset/` — cleaned and original CSV files
- `R_Code/` — complete R analysis script
- `Graphs/` — model evaluation visualizations
- `Outputs/` — statistical test results, metrics and confusion matrix
- `Report/` — Word report

## How to run
1. Download or clone this repository.
2. Open the project folder in RStudio.
3. Open `R_Code/Week3_Statistical_Modeling.R`.
4. Set the working directory to the project root.
5. Run the script. It installs missing packages and writes results to `Outputs/` and `Graphs/`.

## R packages
`tidyverse`, `caret`, `pROC`, `car`, `broom`, `corrplot`

## Author
[Enter your full name]

## Note
The report contains reference results generated from the included data. Execute the R script in RStudio and verify its outputs before final submission. This analysis is predictive and associative; it does not establish causation.
