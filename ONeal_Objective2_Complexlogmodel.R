# Install the necessary packages
install.packages(c("dplyr", "tidyr", "glmnet", "caret", "pROC", "MASS", "car", "polynom", "interactions", "corrplot", "ggplot2"))

# Load the required libraries
library(dplyr)
library(tidyr)
library(glmnet)
library(caret)
library(pROC)
library(MASS)
library(car)
library(polynom)
library(interactions)
library(corrplot)
library(ggplot2)

# Read the dataset 
data = read.csv("/Users/office/Desktop/SMU/Applied Statistics/Project 2/Data/adult.data.csv")

# Trim leading and trailing spaces from all character and factor columns
data[] = lapply(data, function(x) if(is.character(x) | is.factor(x)) trimws(x) else x)

# Removing rows with '?'
data = subset(data, !(workclass == "?" | education == "?" | occupation == "?" | native_country == "?"))

# Mutating 'class' column to be binary (0 for 'under_50k' and 1 for 'over_50k')
data = data %>%
  mutate(class = ifelse(tolower(class) == "under_50k", 0, ifelse(tolower(class) == "over_50k", 1, NA)))

# Define the columns to be treated as factors
factor_columns = c("workclass", "education", "martial_status", "occupation", "relationship", "race", "sex", "native_country")

# Convert these columns to factors
data[factor_columns] = lapply(data[factor_columns], factor)

# Create interaction terms
data$interaction_age_education = data$age * data$education_num
data$age_squared = data$age^2
data$education_num_squared = data$education_num^2

# Split the data into training and testing sets
set.seed(123)  # For reproducibility
trainIndex = createDataPartition(data$class, p = 0.7, list = FALSE)
train_data = data[trainIndex, ]
test_data = data[-trainIndex, ]

# Define formula for the logistic regression model
formula = formula("class ~ age + fnlwgt + education_num + workclass + occupation + relationship + race + sex + capital_gain + capital_loss + hours_per_week + native_country + interaction_age_education + age_squared + education_num_squared")

# Fit the model
X = model.matrix(formula, data = train_data)
y = train_data$class

# Fit the model with cross-validation
set.seed(123)
cv_fit = cv.glmnet(X, y, family = "binomial", alpha = 1)

# Get the lambda that gives the minimum cross-validated error
lambda_min = cv_fit$lambda.min

# Refit the model using this lambda
fit = glmnet(X, y, family = "binomial", alpha = 1, lambda = lambda_min)

# The nonzero coefficients correspond to the selected features
coef(fit)


# Predict on the test set
X_test = model.matrix(formula, data = test_data)
predictions = predict(fit, newx = X_test, type = "response")
predicted_scores = predictions[, 1]

# Create the ROC object
roc_obj = roc(test_data$class, predicted_scores)

# Calculate the AUC
auc_value = auc(roc_obj)
print(paste0("AUC: ", auc_value))

# Convert the predicted scores to class labels (0 or 1) using a threshold (0.5 by default)
predicted_labels = ifelse(predicted_scores >= 0.5, 1, 0)

predicted_labels = as.factor(ifelse(predicted_scores >= 0.5, 1, 0))
levels(predicted_labels) = c(0, 1)

test_data$class = as.factor(test_data$class)
levels(test_data$class) = c(0, 1)

confusion_matrix = confusionMatrix(predicted_labels, test_data$class)


# Create confusion matrix
confusionMatrix = confusionMatrix(predicted_labels, test_data$class, positive = "1")

# Print confusion matrix
print(confusionMatrix)

# Compute metrics
sensitivity = confusionMatrix$byClass["Sensitivity"]
specificity = confusionMatrix$byClass["Specificity"]
ppv = confusionMatrix$byClass["Pos Pred Value"]
npv = confusionMatrix$byClass["Neg Pred Value"]
auroc = auc(roc_obj)

# Print metrics
cat("Sensitivity (Recall): ", sensitivity, "\n")
cat("Specificity: ", specificity, "\n")
cat("Positive Predictive Value (Precision): ", ppv, "\n")
cat("Negative Predictive Value: ", npv, "\n")
cat("Area Under the ROC Curve: ", auroc, "\n")

# Prevalence
prevalence = sum(test_data$class == "1") / length(test_data$class)
cat("Prevalence: ", prevalence, "\n")


