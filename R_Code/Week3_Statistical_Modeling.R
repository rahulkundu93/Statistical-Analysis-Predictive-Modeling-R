# WEEK 3: Statistical Analysis and Predictive Modeling using R
# Project: Titanic Survival Prediction using Logistic Regression
# Run from the Week3_Statistical_Modeling project root.

packages <- c("tidyverse", "caret", "pROC", "car", "broom", "corrplot")
missing <- packages[!packages %in% rownames(installed.packages())]
if(length(missing) > 0) install.packages(missing)
invisible(lapply(packages, library, character.only = TRUE))

dir.create("Outputs", showWarnings = FALSE)
dir.create("Graphs", showWarnings = FALSE)

# 1. Load and inspect
raw <- read.csv("Dataset/Cleaned_Dataset.csv", stringsAsFactors = FALSE)
dat <- raw %>% select(Survived, Pclass, Sex, Age, SibSp, Parch, Fare, Embarked)
# Week 1 cleaned file stores Survived as Yes/No.
dat$Survived <- ifelse(tolower(as.character(dat$Survived)) == "yes", 1, 0)
dat <- dat %>% mutate(
  Survived = as.integer(Survived),
  Pclass = factor(Pclass), Sex = factor(Sex), Embarked = factor(Embarked)
) %>% drop_na()

capture.output(str(dat), summary(dat), file = "Outputs/Initial_Data_Inspection.txt")
write.csv(dat, "Outputs/Modeling_Data.csv", row.names = FALSE)

# 2. Hypothesis testing; significance level alpha = 0.05
# H0-1: Survival is independent of gender.
sex_chi <- chisq.test(table(dat$Sex, dat$Survived))
# H0-2: Survival is independent of passenger class.
class_chi <- chisq.test(table(dat$Pclass, dat$Survived))

# Check age normality by survival group; use Welch t-test if both groups
# are reasonably normal, otherwise use the non-parametric Wilcoxon test.
age_surv <- dat$Age[dat$Survived == 1]
age_no <- dat$Age[dat$Survived == 0]
shapiro_surv <- shapiro.test(sample(age_surv, min(length(age_surv), 5000)))
shapiro_no <- shapiro.test(sample(age_no, min(length(age_no), 5000)))
age_test_name <- "Welch t-test"
if(shapiro_surv$p.value < 0.05 || shapiro_no$p.value < 0.05) {
  age_test <- wilcox.test(Age ~ Survived, data = dat)
  age_test_name <- "Wilcoxon rank-sum test"
} else {
  age_test <- t.test(Age ~ Survived, data = dat)
}
# Fare is usually skewed, so use Wilcoxon rank-sum test.
fare_test <- wilcox.test(Fare ~ Survived, data = dat)

hypothesis_results <- tibble(
  Comparison = c("Gender vs survival", "Passenger class vs survival",
                 "Age vs survival", "Fare vs survival"),
  Test = c("Chi-square", "Chi-square", age_test_name, "Wilcoxon rank-sum"),
  Statistic = c(unname(sex_chi$statistic), unname(class_chi$statistic),
                unname(age_test$statistic), unname(fare_test$statistic)),
  P_value = c(sex_chi$p.value, class_chi$p.value,
              age_test$p.value, fare_test$p.value)
) %>% mutate(Decision = ifelse(P_value < 0.05, "Reject H0", "Fail to reject H0"))
write.csv(hypothesis_results, "Outputs/Hypothesis_Test_Results.csv", row.names = FALSE)
write.csv(tibble(Group = c("Survivors", "Non-survivors"),
                 Shapiro_W = c(unname(shapiro_surv$statistic), unname(shapiro_no$statistic)),
                 P_value = c(shapiro_surv$p.value, shapiro_no$p.value)),
          "Outputs/Age_Normality_Tests.csv", row.names = FALSE)

# Spearman correlations among numeric predictors
cor_data <- dat %>% select(Age, SibSp, Parch, Fare)
cor_matrix <- cor(cor_data, method = "spearman", use = "complete.obs")
write.csv(cor_matrix, "Outputs/Spearman_Correlation_Matrix.csv")
png("Graphs/05_Correlation_Heatmap.png", width = 900, height = 700)
corrplot(cor_matrix, method = "color", type = "upper", addCoef.col = "black")
dev.off()

# 3. Stratified 80/20 split and 5-fold cross-validation
set.seed(42)
index <- createDataPartition(factor(dat$Survived), p = 0.80, list = FALSE)
train_data <- dat[index, ]
test_data <- dat[-index, ]
train_data$Outcome <- factor(ifelse(train_data$Survived == 1, "Yes", "No"),
                             levels = c("Yes", "No"))
test_data$Outcome <- factor(ifelse(test_data$Survived == 1, "Yes", "No"),
                            levels = c("Yes", "No"))
control <- trainControl(method = "cv", number = 5, classProbs = TRUE,
                        summaryFunction = twoClassSummary,
                        savePredictions = "final")
set.seed(42)
model <- train(
  Outcome ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
  data = train_data, method = "glm", family = binomial(),
  trControl = control, metric = "ROC"
)
capture.output(summary(model$finalModel),
               file = "Outputs/Logistic_Regression_Summary.txt")
write.csv(model$results, "Outputs/Cross_Validation_Results.csv", row.names = FALSE)

# 4. Test-set evaluation
prob <- predict(model, newdata = test_data, type = "prob")[, "Yes"]
pred <- factor(ifelse(prob >= 0.50, "Yes", "No"), levels = c("Yes", "No"))
cm <- confusionMatrix(pred, test_data$Outcome, positive = "Yes")
roc_obj <- roc(test_data$Outcome, prob, levels = c("No", "Yes"), direction = "<")
metrics <- tibble(
  Accuracy = unname(cm$overall["Accuracy"]),
  Precision = unname(cm$byClass["Precision"]),
  Recall = unname(cm$byClass["Recall"]),
  F1 = unname(cm$byClass["F1"]),
  ROC_AUC = as.numeric(auc(roc_obj))
)
print(cm); print(metrics)
write.csv(metrics, "Outputs/Performance_Metrics.csv", row.names = FALSE)
capture.output(cm, file = "Outputs/Confusion_Matrix_Summary.txt")
write.csv(broom::tidy(model$finalModel, conf.int = TRUE, exponentiate = TRUE),
          "Outputs/Odds_Ratios.csv", row.names = FALSE)

# 5. Diagnostics
png("Graphs/01_Confusion_Matrix.png", width = 900, height = 700)
fourfoldplot(cm$table, color = c("#D95F02", "#1B9E77"), conf.level = 0,
             margin = 1, main = "Confusion Matrix - Test Set")
dev.off()
png("Graphs/02_ROC_Curve.png", width = 900, height = 700)
plot(roc_obj, col = "blue", lwd = 3,
     main = paste("ROC Curve; AUC =", round(auc(roc_obj), 3)))
abline(a = 0, b = 1, lty = 2, col = "gray")
dev.off()
png("Graphs/03_Model_Diagnostics.png", width = 900, height = 700)
par(mfrow = c(2, 2)); plot(model$finalModel); dev.off()
capture.output(tryCatch(car::vif(model$finalModel),
                        error = function(e) paste("VIF unavailable:", e$message)),
               file = "Outputs/VIF_Results.txt")
cat("Analysis complete. Review the Outputs and Graphs folders.\n")
