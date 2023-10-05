# Load the necessary libraries
library(tidyverse)  # For data manipulation and visualization
library(caret)      # For machine learning and data preprocessing
library(ggplot2)    # For data visualization
library(pROC)       # For ROC curve and AUC
library(corrplot)   # For correlation heatmap

# Read the CSV file
data <- read.csv("Data analysis with R/data.csv")

# Handling Missing Values: Remove rows with NA values
data <- na.omit(data)

# Drop the customerID column as it's not relevant for analysis
data <- data %>% select(-customerID)

# Apply one-hot encoding, omitting one category for each categorical variable to avoid multicollinearity
data_encoded <- model.matrix(~ . - 1, data)
data_encoded <- as.data.frame(data_encoded) # Convert the result back to a data frame
data_encoded$genderFemale <- NULL # Remove the 'genderFemale' column

# Split the dataset into a training set and a testing set using an 80-20 split
set.seed(123) # Set a seed for reproducibility
index <- createDataPartition(data_encoded$ChurnYes, p = 0.8, list = FALSE)
train_data <- data_encoded[index, ]
test_data <- data_encoded[-index, ]

# Visualization: Histograms for numerical columns
hist_cols <- c("tenure", "MonthlyCharges", "TotalCharges")
for (col in hist_cols) {
  p <- ggplot(train_data, aes(x = !!sym(col))) + 
    geom_histogram(fill="blue", color="black", alpha=0.7) + 
    labs(title=paste("Histogram of", col), x=col, y="Frequency") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
  print(p)
}

# Visualization: Box plots of numerical columns against Churn
for (col in hist_cols) {
  p <- ggplot(train_data, aes(x = as.factor(ChurnYes), y = !!sym(col))) + 
    geom_boxplot(fill="blue", color="black", alpha=0.7) + 
    labs(title=paste("Boxplot of", col, "by Churn"), x="ChurnYes", y=col) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
  print(p)
}

# Data Preprocessing: Scale the numerical columns
columns_to_scale <- c("tenure", "MonthlyCharges", "TotalCharges")
preprocess_params <- preProcess(train_data[, columns_to_scale], method = c("range"))
train_data[, columns_to_scale] <- predict(preprocess_params, train_data[, columns_to_scale])
test_data[, columns_to_scale] <- predict(preprocess_params, test_data[, columns_to_scale])

# Visualization: Compute and display the correlation heatmap for the training data
cor_matrix <- cor(train_data[, sapply(train_data, is.numeric)])
corrplot(cor_matrix, method = "color", type = "upper", order = "hclust", tl.cex = 0.7, tl.col = "black", title = "Correlation Heatmap", mar = c(0,0,1,0), addCoef.col = "black", number.cex = 0.5)

# Feature Selection: Drop columns based on correlation analysis
columns_to_remove <- c("MultipleLinesNo phone service", "StreamingMoviesNo internet service", "StreamingTVNo internet service", "TechSupportNo internet service", "DeviceProtectionNo internet service", "OnlineBackupNo internet service", "OnlineSecurityNo internet service", "TotalCharges")
train_data <- train_data %>% select(-one_of(columns_to_remove))
test_data <- test_data %>% select(-one_of(columns_to_remove))

# Convert 'ChurnYes' to a factor
train_data$ChurnYes <- as.factor(train_data$ChurnYes)
test_data$ChurnYes <- as.factor(test_data$ChurnYes)

# Train a logistic regression model using all features
train_control <- trainControl(method = "cv", number = 5)
full_model <- train(form = ChurnYes ~ ., data = train_data, trControl = train_control, method = "glm", family = "binomial")

#View summary of the model
summary(full_model)

# Feature Selection: Select significant features based on p-values
selected_features <- c("tenure", "MultipleLinesYes", "InternetServiceFiber optic", "InternetServiceNo", "StreamingTVYes", "StreamingMoviesYes", "ContractOne year", "ContractTwo year", "PaperlessBillingYes", "PaymentMethodElectronic check", "ChurnYes")
selected_train_data <- train_data[, selected_features]
selected_test_data <- test_data[, selected_features]

# Train the logistic regression model using the selected features
selected_model <- train(form = ChurnYes ~ ., data = selected_train_data, trControl = train_control, method = "glm", family = "binomial")

# Function to evaluate the model based on different metrics
evaluate_model <- function(model, test_data, threshold) {
  # Predictions
  predictions_prob <- predict(model, newdata=test_data, type="prob")[,2]
  predictions <- ifelse(predictions_prob > threshold, 1, 0)
  
  # Compute metrics from the confusion matrix
  conf_matrix <- table(test_data$ChurnYes, predictions)
  accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
  precision <- conf_matrix[2,2] / (conf_matrix[2,2] + conf_matrix[1,2])
  recall <- conf_matrix[2,2] / (conf_matrix[2,2] + conf_matrix[2,1])
  f1_score <- 2 * ((precision * recall) / (precision + recall))
  roc_obj <- roc(test_data$ChurnYes, predictions_prob)
  auc_value <- auc(roc_obj)
  
  # Display metrics
  cat("Threshold:", threshold, "\n")
  cat("Accuracy:", accuracy, "\n")
  cat("Precision:", precision, "\n")
  cat("Recall:", recall, "\n")
  cat("F1-Score:", f1_score, "\n")
  cat("ROC AUC:", auc_value, "\n\n")
}

# Evaluate the model on the training and test data
cat("Training Evaluation:\n")
evaluate_model(selected_model, selected_train_data, 0.5)
cat("Test Evaluation:\n")
evaluate_model(selected_model, selected_test_data, 0.5)

# Display the ROC curve for the test data
predictions_prob <- predict(selected_model, newdata=selected_test_data, type="prob")[,2]
roc_obj <- roc(selected_test_data$ChurnYes, predictions_prob)
plot.roc(roc_obj, legacy.axes=TRUE)

#Evaluate the model using 0.167 as the threshold
cat("Training Evaluation:\n")
evaluate_model(selected_model, selected_train_data, 0.167)
cat("Test Evaluation:\n")
evaluate_model(selected_model, selected_test_data, 0.167)

# Compute the threshold based on class distribution
threshold_ratio <- sum(selected_train_data$ChurnYes == 1) / nrow(selected_train_data)

# Evaluate the model using the computed threshold
cat("Training Evaluation (Threshold based on Class Distribution):\n")
evaluate_model(selected_model, selected_train_data, threshold_ratio)
cat("Test Evaluation (Threshold based on Class Distribution):\n")
evaluate_model(selected_model, selected_test_data, threshold_ratio)
