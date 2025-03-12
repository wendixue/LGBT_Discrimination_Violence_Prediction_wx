#------------------------------------------------------------
# 11) Violence Prediction Modeling Framework
#------------------------------------------------------------

#------------------------------------------------------------
# 11.1) Prepare common modeling data
#------------------------------------------------------------
# Select common set of predictor variables
predictors <- c(
  "gender", "sexuality", "minority", "education", "employed", 
  "econ_level", "marital_status", "health", "religion", "residence", 
  "coming_out_age", "outness_mean", "location_comfort", 
  "disclosure_locations", "country", "age", "genderXmin", "genderXecon"
)

# Create modeling dataset and handle NA values
model_data <- df_filtered[, c(predictors, "violence_group")]
model_data <- na.omit(model_data)  # Remove rows with missing values

# Verify distribution of violence groups
print("Distribution of violence groups in modeling data:")
print(table(model_data$violence_group, useNA = "always"))

# Split into training and test sets
set.seed(123)
if (requireNamespace("caret", quietly = TRUE)) {
  library(caret)
  train_index <- createDataPartition(model_data$violence_group, p = 0.7, list = FALSE)
} else {
  # Manual alternative if caret is not available
  n <- nrow(model_data)
  train_index <- sample(1:n, size = round(0.7 * n))
}
train_data <- model_data[train_index, ]
test_data <- model_data[-train_index, ]

# Check distribution in training set
print("Violence group distribution in training data:")
print(table(train_data$violence_group))

#------------------------------------------------------------
# 11.2) Create balanced training dataset using downsampling
#------------------------------------------------------------
# Find minimum class count
min_class_count <- min(table(train_data$violence_group))
print(paste("Minimum class count:", min_class_count))

# Create balanced training set through downsampling
set.seed(456)
balanced_train <- do.call(rbind, lapply(levels(train_data$violence_group), function(class_level) {
  class_data <- train_data[train_data$violence_group == class_level, ]
  if (nrow(class_data) > min_class_count) {
    return(class_data[sample(1:nrow(class_data), min_class_count), ])
  } else {
    return(class_data)
  }
}))

# Verify balanced distribution
print("Distribution after balancing:")
print(table(balanced_train$violence_group))

#------------------------------------------------------------
# 11.3) Random Forest Model with balanced data
#------------------------------------------------------------
library(ranger) # Fast implementation of Random Forest

# Function to evaluate RF performance with different numbers of trees
evaluate_tree_performance <- function(data, max_trees=500, step=50) {
  # Create vectors to store results
  tree_counts <- seq(step, max_trees, by=step)
  error_rates <- numeric(length(tree_counts))
  
  # For each tree count
  for(i in seq_along(tree_counts)) {
    num_trees <- tree_counts[i]
    cat("Training with", num_trees, "trees...\n")
    
    # Train model with OOB error enabled
    model <- ranger(
      violence_group ~ .,
      data = data,
      num.trees = num_trees,
      mtry = ceiling(length(predictors)/3),
      min.node.size = 1,
      probability = TRUE,
      oob.error = TRUE,
      respect.unordered.factors = "order"
    )
    
    # Record OOB error rate
    error_rates[i] <- model$prediction.error
  }
  
  # Return results
  return(data.frame(Trees = tree_counts, Error = error_rates))
}

# Evaluate tree performance
set.seed(789)
rf_error_curve <- evaluate_tree_performance(balanced_train, max_trees=1000, step=50)

# Plot error curve
library(ggplot2)
error_plot <- ggplot(rf_error_curve, aes(x=Trees, y=Error)) +
  geom_line() +
  geom_point() +
  theme_minimal() +
  labs(
    title = "Random Forest Error Rate vs Number of Trees",
    x = "Number of Trees",
    y = "Out-of-Bag Error Rate"
  )
print(error_plot)

# Find optimal number of trees
optimal_trees <- rf_error_curve$Trees[which.min(rf_error_curve$Error)]
print(paste("Optimal number of trees:", optimal_trees))

# Fit final random forest with optimal number of trees
set.seed(789)
rf_model <- ranger(
  violence_group ~ ., 
  data = balanced_train,
  num.trees = optimal_trees,
  mtry = ceiling(length(predictors)/3),
  min.node.size = 1,
  importance = "impurity",
  probability = TRUE,
  respect.unordered.factors = "order"
)

# Make predictions on test data
rf_pred_probs <- predict(rf_model, data = test_data)
rf_predictions <- factor(
  colnames(rf_pred_probs$predictions)[apply(rf_pred_probs$predictions, 1, which.max)],
  levels = levels(test_data$violence_group)
)

# Create confusion matrix
library(caret)
rf_conf_matrix <- confusionMatrix(rf_predictions, test_data$violence_group)
print("Random Forest Performance:")
print(rf_conf_matrix)

# Extract variable importance
rf_importance <- rf_model$variable.importance
rf_importance_df <- data.frame(
  Variable = names(rf_importance),
  Importance = rf_importance
) |>
  arrange(desc(Importance))

# Print top important features
print("Random Forest - Top 15 Important Features:")
print(head(rf_importance_df, 15))

#------------------------------------------------------------
# 11.4) XGBoost Model with balanced data
#------------------------------------------------------------
library(xgboost)
library(Matrix)

# Prepare data for XGBoost (convert categorical variables)
x_train <- model.matrix(~ . - violence_group, data = balanced_train)[, -1] # Remove intercept
x_test <- model.matrix(~ . - violence_group, data = test_data)[, -1]

# Convert target to numeric (0-based)
y_train <- as.integer(balanced_train$violence_group) - 1
y_test <- as.integer(test_data$violence_group) - 1

# Create DMatrix objects
dtrain <- xgb.DMatrix(data = x_train, label = y_train)
dtest <- xgb.DMatrix(data = x_test, label = y_test)

# Calculate class weights for imbalance
class_weights <- table(balanced_train$violence_group)
class_weights <- sum(class_weights) / (length(class_weights) * class_weights)

# Create sample weight vector
sample_weights <- numeric(length(y_train))
for (i in 1:length(y_train)) {
  sample_weights[i] <- class_weights[y_train[i] + 1]
}
setinfo(dtrain, "weight", sample_weights)

# Set XGBoost parameters
xgb_params <- list(
  objective = "multi:softprob",
  eval_metric = "mlogloss",
  num_class = length(levels(balanced_train$violence_group)),
  eta = 0.05,
  max_depth = 6,
  min_child_weight = 1,
  subsample = 0.8,
  colsample_bytree = 0.8,
  gamma = 0.1,
  tree_method = "hist"
)

# Find optimal number of rounds using cross-validation
set.seed(456)
cv_results <- xgb.cv(
  params = xgb_params,
  data = dtrain,
  nrounds = 1000,
  nfold = 5,
  early_stopping_rounds = 50,
  verbose = 0
)

# Plot learning curve
cv_data <- data.frame(
  Iteration = 1:length(cv_results$evaluation_log$test_mlogloss_mean),
  Train_Error = cv_results$evaluation_log$train_mlogloss_mean,
  Test_Error = cv_results$evaluation_log$test_mlogloss_mean
)

library(tidyr)
cv_data_long <- pivot_longer(
  cv_data,
  cols = c("Train_Error", "Test_Error"),
  names_to = "Dataset",
  values_to = "Error"
)

xgb_learning_curve <- ggplot(cv_data_long, aes(x = Iteration, y = Error, color = Dataset)) +
  geom_line() +
  geom_vline(xintercept = which.min(cv_results$evaluation_log$test_mlogloss_mean), 
             linetype = "dashed", color = "black") +
  theme_minimal() +
  labs(title = "XGBoost Learning Curve",
       x = "Number of Boosting Rounds",
       y = "Multi-class Log Loss")

print(xgb_learning_curve)

best_nrounds <- which.min(cv_results$evaluation_log$test_mlogloss_mean)
print(paste("Best number of rounds:", best_nrounds))

# Train final XGBoost model
set.seed(789)
xgb_model <- xgb.train(
  params = xgb_params,
  data = dtrain,
  nrounds = best_nrounds,
  watchlist = list(train = dtrain, test = dtest),
  verbose = 0
)

# Make predictions
xgb_pred_probs <- predict(xgb_model, dtest, reshape = TRUE)
xgb_predictions <- factor(
  levels(test_data$violence_group)[apply(xgb_pred_probs, 1, which.max)],
  levels = levels(test_data$violence_group)
)

# Create confusion matrix
xgb_conf_matrix <- confusionMatrix(xgb_predictions, test_data$violence_group)
print("XGBoost Performance:")
print(xgb_conf_matrix)

# Get feature importance
xgb_importance <- xgb.importance(feature_names = colnames(x_train), model = xgb_model)
xgb_importance_df <- as.data.frame(xgb_importance) |>
  arrange(desc(Gain))

# Print top important features
print("XGBoost - Top 15 Important Features:")
print(head(xgb_importance_df, 15))


#------------------------------------------------------------
# 11.5) Model Comparison
#------------------------------------------------------------
# Function to calculate average class-specific metrics
calc_avg_metrics <- function(conf_matrix) {
  classes <- rownames(conf_matrix$byClass)
  metrics <- c("Sensitivity", "Specificity", "Pos Pred Value", "F1")
  
  result <- sapply(metrics, function(metric) {
    mean(sapply(classes, function(cls) conf_matrix$byClass[cls, metric]))
  })
  
  return(result)
}

# Calculate metrics
rf_avg <- calc_avg_metrics(rf_conf_matrix)
xgb_avg <- calc_avg_metrics(xgb_conf_matrix)

# Create comparison dataframe
model_metrics <- data.frame(
  Model = c("Random Forest", "XGBoost"),
  Accuracy = c(
    rf_conf_matrix$overall["Accuracy"],
    xgb_conf_matrix$overall["Accuracy"]
  ),
  Kappa = c(
    rf_conf_matrix$overall["Kappa"],
    xgb_conf_matrix$overall["Kappa"]
  ),
  Sensitivity = c(
    rf_avg["Sensitivity"],
    xgb_avg["Sensitivity"]
  ),
  Specificity = c(
    rf_avg["Specificity"],
    xgb_avg["Specificity"]
  ),
  Precision = c(
    rf_avg["Pos Pred Value"],
    xgb_avg["Pos Pred Value"]
  ),
  F1 = c(
    rf_avg["F1"],
    xgb_avg["F1"]
  )
)

# Print model comparison
print("Model Performance Comparison:")
print(model_metrics)

# Create comparison visualization
if (requireNamespace("ggplot2", quietly = TRUE) && 
    requireNamespace("tidyr", quietly = TRUE)) {
  # Convert to long format for plotting
  comparison_long <- tidyr::pivot_longer(
    model_metrics,
    cols = c("Accuracy", "Kappa", "Sensitivity", "Specificity", "Precision", "F1"),
    names_to = "Metric",
    values_to = "Value"
  )
  
  # Create bar chart
  model_comp_plot <- ggplot(comparison_long, aes(x = Metric, y = Value, fill = Model)) +
    geom_bar(stat = "identity", position = "dodge") +
    theme_minimal() +
    labs(title = "Model Performance Comparison",
         y = "Score",
         x = "Metric") +
    scale_fill_brewer(palette = "Set1") +
    coord_flip()
  
  print(model_comp_plot)
}

#------------------------------------------------------------
# 11.6) XGBoost Parameter Tuning
#------------------------------------------------------------
# Define parameter grid
param_grid <- expand.grid(
  eta = c(0.01, 0.05, 0.1),          # Learning rate
  max_depth = c(3, 5, 7),            # Maximum tree depth
  min_child_weight = c(1, 3, 5),        # Minimum sum of instance weight
  subsample = 0.8,                   # Subsample ratio (fixed)
  colsample_bytree = 0.8,            # Feature sampling ratio (fixed)
  gamma = 0.1                        # Minimum loss reduction (fixed)
)

print("Parameter tuning grid:")
print(param_grid)

# Initialize variables for best parameters
best_params <- NULL
best_score <- 0
best_nrounds <- 0
cv_results_list <- list()

# Iterate through parameter grid
for (i in 1:nrow(param_grid)) {
  # Get current parameter combination
  current_params <- list(
    objective = "multi:softprob",
    eval_metric = "mlogloss",
    num_class = length(levels(balanced_train$violence_group)),
    eta = param_grid$eta[i],
    max_depth = param_grid$max_depth[i],
    min_child_weight = param_grid$min_child_weight[i],
    subsample = param_grid$subsample[i],
    colsample_bytree = param_grid$colsample_bytree[i],
    gamma = param_grid$gamma[i],
    tree_method = "hist"
  )
  
  # Print current combination
  cat("Testing parameter combination ", i, "/", nrow(param_grid), ":\n")
  print(as.data.frame(param_grid[i, ]))
  
  # Cross-validation
  set.seed(456)
  cv_result <- xgb.cv(
    params = current_params,
    data = dtrain,
    nrounds = 1000,
    nfold = 5,
    early_stopping_rounds = 50,
    verbose = 0
  )
  
  # Store results
  cv_results_list[[i]] <- cv_result
  
  # Get best round and score
  best_iter <- which.min(cv_result$evaluation_log$test_mlogloss_mean)
  current_score <- 1 - cv_result$evaluation_log$test_mlogloss_mean[best_iter]  # Convert to score
  
  # Print current results
  cat("  Best iteration:", best_iter, "\n")
  cat("  Cross-validation score:", current_score, "\n\n")
  
  # Update best parameters
  if (current_score > best_score) {
    best_score <- current_score
    best_params <- current_params
    best_nrounds <- best_iter
  }
}

# Print best parameters
print("Best parameter combination:")
print(best_params)
print(paste("Best number of rounds:", best_nrounds))
print(paste("Best score:", best_score))

# Train final model with best parameters
set.seed(789)
xgb_model_tuned <- xgb.train(
  params = best_params,
  data = dtrain,
  nrounds = best_nrounds,
  watchlist = list(train = dtrain, test = dtest),
  verbose = 0
)

# Make predictions
xgb_pred_probs_tuned <- predict(xgb_model_tuned, dtest, reshape = TRUE)
xgb_predictions_tuned <- factor(
  levels(test_data$violence_group)[apply(xgb_pred_probs_tuned, 1, which.max)],
  levels = levels(test_data$violence_group)
)

# Create confusion matrix
xgb_conf_matrix_tuned <- confusionMatrix(xgb_predictions_tuned, test_data$violence_group)
print("Tuned XGBoost model performance:")
print(xgb_conf_matrix_tuned)

# Calculate performance improvement
print("Performance improvement:")
print(paste("Accuracy improvement:", 
            round((xgb_conf_matrix_tuned$overall["Accuracy"] - 
                     xgb_conf_matrix$overall["Accuracy"]) * 100, 2), 
            "%"))
print(paste("Kappa improvement:", 
            round((xgb_conf_matrix_tuned$overall["Kappa"] - 
                     xgb_conf_matrix$overall["Kappa"]) * 100, 2), 
            "%"))

# Get feature importance for tuned model
xgb_importance_tuned <- xgb.importance(feature_names = colnames(x_train), model = xgb_model_tuned)
xgb_importance_df_tuned <- as.data.frame(xgb_importance_tuned) |>
  arrange(desc(Gain))

# Print feature importance
print("Feature importance in tuned model (top 30):")
print(head(xgb_importance_df_tuned, 30))

#------------------------------------------------------------
# 11.7) Findings and Predictions
#------------------------------------------------------------

## (1) Prediction of violence level at different economic levels

# Load necessary libraries
library(dplyr)
library(ggplot2)
library(tidyr)

# Get all possible values of economic level
econ_levels <- levels(train_data$econ_level)
print("Economic level categories:")
print(econ_levels)

# Create a dataframe to store prediction results for different economic levels
results <- data.frame(
  EconomicLevel = character(),
  Severe_Violence_Prob = numeric(),
  Mild_Violence_Prob = numeric(),
  Never_Violence_Prob = numeric(),
  stringsAsFactors = FALSE
)

# Generate predictions for each economic level
for (econ_category in econ_levels) {
  # Randomly select samples from training data where econ_level equals the current category
  category_samples <- train_data %>%
    filter(econ_level == econ_category) %>%
    sample_n(min(50, sum(train_data$econ_level == econ_category)))
  
  if (nrow(category_samples) == 0) {
    next  # Skip categories with no samples
  }
  
  # Convert these samples to the format required by the model
  category_matrix <- model.matrix(~ . - violence_group, data = category_samples)[, -1]
  
  # Make predictions
  pred_probs <- predict(xgb_model_tuned, as.matrix(category_matrix), reshape = TRUE)
  
  # Calculate average prediction probabilities for all violence categories
  avg_probs <- colMeans(pred_probs)
  
  # Add to results (assuming 3 classes: Never, Mild, Severe in that order)
  results <- rbind(results, data.frame(
    EconomicLevel = econ_category,
    Never_Violence_Prob = avg_probs[1],  # First class - Never
    Mild_Violence_Prob = avg_probs[2],   # Second class - Mild
    Severe_Violence_Prob = avg_probs[3]  # Third class - Severe
  ))
}

# Keep original economic level order (from most difficult to easiest)
results$EconomicLevel <- factor(results$EconomicLevel, levels = econ_levels)

# Convert to long format for plotting
results_long <- results %>%
  pivot_longer(
    cols = c("Never_Violence_Prob", "Mild_Violence_Prob", "Severe_Violence_Prob"),
    names_to = "ViolenceClass",
    values_to = "Probability"
  )

# Set factor levels for violence classes for better ordering
results_long$ViolenceClass <- factor(results_long$ViolenceClass,
                                     levels = c("Severe_Violence_Prob", "Mild_Violence_Prob", "Never_Violence_Prob"),
                                     labels = c("Severe", "Mild", "Never"))

# Create stacked bar chart
p_stacked <- ggplot(results_long, aes(x = EconomicLevel, y = Probability, fill = ViolenceClass)) +
  geom_bar(stat = "identity", position = "stack") +
  scale_fill_manual(values = c("Severe" = "#E41A1C", "Mild" = "#377EB8", "Never" = "#4DAF4A")) +
  labs(
    title = "Predicted Violence Probabilities by Economic Level",
    x = "Economic Level",
    y = "Probability",
    fill = "Violence Level"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14),
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "right"
  )

# Print stacked bar chart
print(p_stacked)

# Save chart
ggsave("economic_level_violence_stacked.png", p_stacked, width = 12, height = 6, dpi = 300)

# Print severe violence probabilities specifically
print("Severe Violence Probabilities by Economic Level:")
print(results %>% select(EconomicLevel, Severe_Violence_Prob) %>% arrange(EconomicLevel))

# Create a line chart for severe violence probability across economic levels
p_line <- ggplot(results, aes(x = EconomicLevel, y = Severe_Violence_Prob, group = 1)) +
  geom_line(size = 1.2, color = "#E41A1C") +
  geom_point(size = 3, color = "#E41A1C") +
  labs(
    title = "Severe Violence Probability Trend Across Economic Levels",
    x = "Economic Level",
    y = "Probability of Severe Violence"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

# Print line chart
print(p_line)

## (1) Prediction of violence level at different minority status
# Load necessary libraries
library(dplyr)
library(ggplot2)
library(tidyr)

# Get all possible values of minority
minority_levels <- levels(train_data$minority)
print("Minority categories:")
print(minority_levels)

# Create a dataframe to store prediction results for different minority categories
results <- data.frame(
  Minority = character(),
  Severe_Violence_Prob = numeric(),
  Mild_Violence_Prob = numeric(),
  Never_Violence_Prob = numeric(),
  stringsAsFactors = FALSE
)

# Generate predictions for each minority category
for (minority_category in minority_levels) {
  # Randomly select samples from training data where minority equals the current category
  category_samples <- train_data %>%
    filter(minority == minority_category) %>%
    sample_n(min(50, sum(train_data$minority == minority_category)))
  
  if (nrow(category_samples) == 0) {
    next  # Skip categories with no samples
  }
  
  # Convert these samples to the format required by the model
  category_matrix <- model.matrix(~ . - violence_group, data = category_samples)[, -1]
  
  # Make predictions
  pred_probs <- predict(xgb_model_tuned, as.matrix(category_matrix), reshape = TRUE)
  
  # Calculate average prediction probabilities for all violence categories
  avg_probs <- colMeans(pred_probs)
  
  # Add to results (assuming 3 classes: Never, Mild, Severe in that order)
  results <- rbind(results, data.frame(
    Minority = minority_category,
    Never_Violence_Prob = avg_probs[1],  # First class - Never
    Mild_Violence_Prob = avg_probs[2],   # Second class - Mild
    Severe_Violence_Prob = avg_probs[3]  # Third class - Severe
  ))
}

# Sort results by Severe Violence probability
results <- results %>% arrange(desc(Severe_Violence_Prob))

# Create factor with ordered levels based on severe violence probability
results$Minority <- factor(results$Minority, levels = results$Minority)

# Convert to long format for plotting
results_long <- results %>%
  pivot_longer(
    cols = c("Never_Violence_Prob", "Mild_Violence_Prob", "Severe_Violence_Prob"),
    names_to = "ViolenceClass",
    values_to = "Probability"
  )

# Set factor levels for violence classes for better ordering
results_long$ViolenceClass <- factor(results_long$ViolenceClass,
                                     levels = c("Severe_Violence_Prob", "Mild_Violence_Prob", "Never_Violence_Prob"),
                                     labels = c("Severe", "Mild", "Never"))

# Create stacked bar chart
p_stacked <- ggplot(results_long, aes(x = Minority, y = Probability, fill = ViolenceClass)) +
  geom_bar(stat = "identity", position = "stack") +
  scale_fill_manual(values = c("Severe" = "#E41A1C", "Mild" = "#377EB8", "Never" = "#4DAF4A")) +
  labs(
    title = "Predicted Violence Probabilities by Minority Category",
    x = "Minority Category",
    y = "Probability",
    fill = "Violence Level"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14),
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "right"
  )

# Print stacked bar chart
print(p_stacked)

## (3) Prediction of violence level at different gender
# Get all possible values of gender
gender_levels <- levels(train_data$gender)
print("Gender categories:")
print(gender_levels)

# Create a dataframe to store prediction results for different gender categories
results <- data.frame(
  Gender = character(),
  Severe_Violence_Prob = numeric(),
  Mild_Violence_Prob = numeric(),
  Never_Violence_Prob = numeric(),
  stringsAsFactors = FALSE
)

# Generate predictions for each gender category
for (gender_category in gender_levels) {
  # Randomly select samples from training data where gender equals the current category
  category_samples <- train_data %>%
    filter(gender == gender_category) %>%
    sample_n(min(50, sum(train_data$gender == gender_category)))
  
  if (nrow(category_samples) == 0) {
    next  # Skip categories with no samples
  }
  
  # Convert these samples to the format required by the model
  category_matrix <- model.matrix(~ . - violence_group, data = category_samples)[, -1]
  
  # Make predictions
  pred_probs <- predict(xgb_model_tuned, as.matrix(category_matrix), reshape = TRUE)
  
  # Calculate average prediction probabilities for all violence categories
  avg_probs <- colMeans(pred_probs)
  
  # Add to results (assuming 3 classes: Never, Mild, Severe in that order)
  results <- rbind(results, data.frame(
    Gender = gender_category,
    Never_Violence_Prob = avg_probs[1],  # First class - Never
    Mild_Violence_Prob = avg_probs[2],   # Second class - Mild
    Severe_Violence_Prob = avg_probs[3]  # Third class - Severe
  ))
}

# Sort results by Severe Violence probability
results <- results %>% arrange(desc(Severe_Violence_Prob))

# Create factor with ordered levels based on severe violence probability
results$Gender <- factor(results$Gender, levels = results$Gender)

# Convert to long format for plotting
results_long <- results %>%
  pivot_longer(
    cols = c("Never_Violence_Prob", "Mild_Violence_Prob", "Severe_Violence_Prob"),
    names_to = "ViolenceClass",
    values_to = "Probability"
  )

# Set factor levels for violence classes for better ordering
results_long$ViolenceClass <- factor(results_long$ViolenceClass,
                                     levels = c("Severe_Violence_Prob", "Mild_Violence_Prob", "Never_Violence_Prob"),
                                     labels = c("Severe", "Mild", "Never"))

# Create stacked bar chart
p_stacked <- ggplot(results_long, aes(x = Gender, y = Probability, fill = ViolenceClass)) +
  geom_bar(stat = "identity", position = "stack") +
  scale_fill_manual(values = c("Severe" = "#E41A1C", "Mild" = "#377EB8", "Never" = "#4DAF4A")) +
  labs(
    title = "Predicted Violence Probabilities by Gender",
    x = "Gender",
    y = "Probability",
    fill = "Violence Level"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14),
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "right"
  )

# Print stacked bar chart
print(p_stacked)

# Save chart
ggsave("gender_violence_stacked.png", p_stacked, width = 10, height = 6, dpi = 300)

# Print results
print("Results sorted by Severe Violence Probability:")
print(results)

