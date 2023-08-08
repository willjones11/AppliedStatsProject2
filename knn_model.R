install.packages(c("class", "caret", "pROC"))
library(class)
library(caret)
library(pROC) 

# Load data
data = read.csv("/Users/office/Desktop/SMU/Applied Statistics/Project 2/Data/adult.data.csv", 
                na.strings = "?", stringsAsFactors = FALSE)

#Clean the data
# Trim leading spaces for character (categorical) columns
for(col in colnames(data)) {
  if(is.character(data[[col]])) {
    data[[col]] = trimws(data[[col]])
  }
}

# Remove rows with NA values
data = data[complete.cases(data), ]

# Convert the class variable to a factor
data$class = as.factor(data$class)

# Split data into training and test sets
set.seed(123)
trainIndex = caret::createDataPartition(data$class, p = .7, list = FALSE)
train = data[trainIndex,]
test = data[-trainIndex,]

# Remove zero-variance predictors
nzv = nearZeroVar(train, saveMetrics = TRUE)
train = train[, !nzv$nzv]
test = test[, !nzv$nzv]

# Define trainControl with AUC as the metric
trControl = trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = twoClassSummary)

# Use tuneLength to tune k
model = train(
  class ~ .,
  data = train,
  method = "knn",
  trControl = trControl,
  tuneLength = 10,  # Or a different number
  preProcess = c("center", "scale"),
  metric = "ROC"
)

# Print model to get the results
print(model)

# Predict on test set
predicted_class_probs = predict(model, newdata = test, type = "prob")
predicted_class = ifelse(predicted_class_probs[,2] > 0.5, levels(train$class)[2], levels(train$class)[1])  # Threshold can be adjusted

# Convert the predicted and actual class to factor with same levels
predicted_class = factor(predicted_class, levels = levels(test$class))
test$class = factor(test$class, levels = levels(predicted_class))

# Evaluate the performance of the model
cm = confusionMatrix(predicted_class, test$class)

# Extract Sensitivity, Specificity, PPV and NPV
sensitivity = cm$byClass["Sensitivity"][1]
specificity = cm$byClass["Specificity"][1]

cat("Sensitivity:", sensitivity, "\n")
cat("Specificity:", specificity, "\n")

#NPV and PPV
TP = cm$table[1, 1]
FP = cm$table[1, 2]
TN = cm$table[2, 2]
FN = cm$table[2, 1]
ppv_manual = TP / (TP + FP)
npv_manual = TN / (TN + FN)
cat("Manually computed PPV:", ppv_manual, "\n")
cat("Manually computed NPV:", npv_manual, "\n")

# Compute and print prevalence
prevalence = sum(test$class == levels(test$class)[2]) / length(test$class)
cat("Prevalence:", prevalence, "\n")

# Compute AUROC
roc_obj = roc(test$class, predicted_class_probs[,2], levels = rev(levels(test$class))) # Specify the order of factor levels for correct computation
auroc = auc(roc_obj)
cat("AUROC:", auroc, "\n")

