#------------------------------------------------------------
# 12) Discrimination Score (disc_total) Prediction Model Framework
#------------------------------------------------------------

#------------------------------------------------------------
# 12.1) Prepare Common Modeling Data
#------------------------------------------------------------
# Select a common set of predictor variables
predictors <- c(
  "gender", "sexuality", "minority", "education", "employed", 
  "econ_level", "marital_status", "health", "religion", "residence", 
  "coming_out_age", "outness_mean", "location_comfort", 
  "disclosure_locations", "country", "age", "genderXmin", "genderXecon"
)

# Create modeling dataset and handle NA values
disc_model_data <- df_filtered[, c(predictors, "disc_total")]
disc_model_data <- na.omit(disc_model_data)  # Remove rows with missing values

# Check distribution of disc_total
print("Distribution statistics for disc_total variable:")
print(summary(disc_model_data$disc_total))

# Check skewness
skewness <- moments::skewness(disc_model_data$disc_total)
print(paste("Skewness of disc_total:", skewness))

# Plot distribution histogram
hist_plot <- ggplot(disc_model_data, aes(x = disc_total)) +
  geom_histogram(bins = 10, fill = "cornflowerblue", color = "black") +
  labs(title = "Distribution of Discrimination Total Score",
       x = "Discrimination Total Score",
       y = "Frequency") +
  theme_minimal()
print(hist_plot)

# Split training and test sets
set.seed(123)
if (requireNamespace("caret", quietly = TRUE)) {
  library(caret)
  train_index <- createDataPartition(disc_model_data$disc_total, p = 0.7, list = FALSE)
} else {
  # Manual split if caret is unavailable
  n <- nrow(disc_model_data)
  train_index <- sample(1:n, size = round(0.7 * n))
}
train_data <- disc_model_data[train_index, ]
test_data <- disc_model_data[-train_index, ]

#------------------------------------------------------------
# 12.2) Data Preprocessing
#------------------------------------------------------------
# Standardize continuous variables
preproc_vars <- c("age", "coming_out_age", "outness_mean", "location_comfort", "disclosure_locations")

# Create preprocessor
if (requireNamespace("caret", quietly = TRUE)) {
  preProc <- preProcess(train_data[, preproc_vars], method = c("center", "scale"))
  train_data[, preproc_vars] <- predict(preProc, train_data[, preproc_vars])
  test_data[, preproc_vars] <- predict(preProc, test_data[, preproc_vars])
} else {
  # Manual standardization
  for (var in preproc_vars) {
    mean_val <- mean(train_data[[var]], na.rm = TRUE)
    sd_val <- sd(train_data[[var]], na.rm = TRUE)
    train_data[[var]] <- (train_data[[var]] - mean_val) / sd_val
    test_data[[var]] <- (test_data[[var]] - mean_val) / sd_val
  }
}

#------------------------------------------------------------
# 12.4) LASSO Model
#------------------------------------------------------------
library(glmnet)

# Prepare data matrices for LASSO model
x_train <- model.matrix(~ . - disc_total, data = train_data)[, -1]  # Remove intercept
x_test <- model.matrix(~ . - disc_total, data = test_data)[, -1]
y_train <- train_data$disc_total
y_test <- test_data$disc_total

# Cross-validation to find optimal lambda
set.seed(456)
cv_lasso <- cv.glmnet(
  x = x_train,
  y = y_train,
  alpha = 1,  # LASSO
  nfolds = 10
)

# Identify best lambda
best_lambda <- cv_lasso$lambda.min
print(paste("Best lambda:", best_lambda))

# Fit model using optimal lambda
lasso_model <- glmnet(
  x = x_train,
  y = y_train,
  alpha = 1,
  lambda = best_lambda
)

# Display coefficients
lasso_coef <- coef(lasso_model)
print("LASSO Non-zero Coefficients:")
print(lasso_coef[lasso_coef[,1] != 0, ])

# Predict
lasso_preds <- predict(lasso_model, newx = x_test)

# Compute evaluation metrics
lasso_mse <- mean((y_test - lasso_preds)^2)
lasso_rmse <- sqrt(lasso_mse)
lasso_mae <- mean(abs(y_test - lasso_preds))
lasso_r2 <- 1 - sum((y_test - lasso_preds)^2) / sum((y_test - mean(y_test))^2)

print("LASSO Performance:")
print(paste("RMSE:", round(lasso_rmse, 4)))
print(paste("MAE:", round(lasso_mae, 4)))
print(paste("R²:", round(lasso_r2, 4)))

# Plot LASSO regularization path
plot(cv_lasso)

#------------------------------------------------------------
# 12.5) Random Forest Model
#------------------------------------------------------------
library(ranger)  # Fast Random Forest implementation

# Train Random Forest model
set.seed(789)
rf_model <- ranger(
  disc_total ~ .,
  data = train_data,
  num.trees = 500,
  mtry = ceiling(length(predictors)/3),
  min.node.size = 5,
  importance = "impurity"
)

# Predict on test data
rf_preds <- predict(rf_model, data = test_data)$predictions

# Compute evaluation metrics
rf_mse <- mean((test_data$disc_total - rf_preds)^2)
rf_rmse <- sqrt(rf_mse)
rf_mae <- mean(abs(test_data$disc_total - rf_preds))
rf_r2 <- 1 - sum((test_data$disc_total - rf_preds)^2) / 
  sum((test_data$disc_total - mean(test_data$disc_total))^2)

print("Random Forest Performance:")
print(paste("RMSE:", round(rf_rmse, 4)))
print(paste("MAE:", round(rf_mae, 4)))
print(paste("R²:", round(rf_r2, 4)))

# Extract feature importance
rf_importance <- rf_model$variable.importance
rf_importance_df <- data.frame(
  Variable = names(rf_importance),
  Importance = rf_importance
) |>
  arrange(desc(Importance))

# Print top features
print("Random Forest - Top 15 Important Features:")
print(head(rf_importance_df, 15))

---------------------------------------------------------
# 12.6) XGBoost Model
#------------------------------------------------------------
library(xgboost)
library(Matrix)

# Prepare XGBoost data (convert categorical variables)
x_train_xgb <- model.matrix(~ . - disc_total, data = train_data)[, -1]  # Remove intercept
x_test_xgb <- model.matrix(~ . - disc_total, data = test_data)[, -1]

# Create DMatrix objects
dtrain <- xgb.DMatrix(data = x_train_xgb, label = train_data$disc_total)
dtest <- xgb.DMatrix(data = x_test_xgb, label = test_data$disc_total)

# Set XGBoost parameters
xgb_params <- list(
  objective = "reg:squarederror",
  eval_metric = "rmse",
  eta = 0.05,
  max_depth = 6,
  min_child_weight = 1,
  subsample = 0.8,
  colsample_bytree = 0.8,
  gamma = 0.1,
  tree_method = "hist"
)

# Use cross-validation to find the optimal number of rounds
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
  Iteration = 1:length(cv_results$evaluation_log$test_rmse_mean),
  Train_Error = cv_results$evaluation_log$train_rmse_mean,
  Test_Error = cv_results$evaluation_log$test_rmse_mean
)

cv_data_long <- tidyr::pivot_longer(
  cv_data,
  cols = c("Train_Error", "Test_Error"),
  names_to = "Dataset",
  values_to = "Error"
)

xgb_learning_curve <- ggplot(cv_data_long, aes(x = Iteration, y = Error, color = Dataset)) +
  geom_line() +
  geom_vline(xintercept = which.min(cv_results$evaluation_log$test_rmse_mean), 
             linetype = "dashed", color = "black") +
  theme_minimal() +
  labs(title = "XGBoost Learning Curve",
       x = "Number of Boosting Rounds",
       y = "Root Mean Squared Error")

print(xgb_learning_curve)

best_nrounds <- which.min(cv_results$evaluation_log$test_rmse_mean)
print(paste("Best number of rounds:", best_nrounds))

# Train the final XGBoost model
set.seed(789)
xgb_model <- xgb.train(
  params = xgb_params,
  data = dtrain,
  nrounds = best_nrounds,
  watchlist = list(train = dtrain, test = dtest),
  verbose = 0
)

# Predictions
xgb_preds <- predict(xgb_model, dtest)

# Compute evaluation metrics
xgb_mse <- mean((test_data$disc_total - xgb_preds)^2)
xgb_rmse <- sqrt(xgb_mse)
xgb_mae <- mean(abs(test_data$disc_total - xgb_preds))
xgb_r2 <- 1 - sum((test_data$disc_total - xgb_preds)^2) / 
  sum((test_data$disc_total - mean(test_data$disc_total))^2)

print("XGBoost Performance:")
print(paste("RMSE:", round(xgb_rmse, 4)))
print(paste("MAE:", round(xgb_mae, 4)))
print(paste("R²:", round(xgb_r2, 4)))

# Get feature importance
xgb_importance <- xgb.importance(feature_names = colnames(x_train_xgb), model = xgb_model)
xgb_importance_df <- as.data.frame(xgb_importance) |>
  arrange(desc(Gain))

# Print top important features
print("XGBoost - Top 15 Important Features:")
print(head(xgb_importance_df, 15))

#------------------------------------------------------------
# 12.7) Model Comparison
#------------------------------------------------------------
# Create a comparison dataframe
model_metrics <- data.frame(
  Model = c("LASSO", "Random Forest", "XGBoost"),
  RMSE = c(
    lasso_rmse,
    rf_rmse,
    xgb_rmse
  ),
  MAE = c(
    lasso_mae,
    rf_mae,
    xgb_mae
  ),
  R_Squared = c(
    lasso_r2,
    rf_r2,
    xgb_r2
  )
)

# Print model comparison
print("Model Performance Comparison:")
print(model_metrics)

# Create visualization for comparison
if (requireNamespace("ggplot2", quietly = TRUE) && 
    requireNamespace("tidyr", quietly = TRUE)) {
  # Convert to long format for plotting
  comparison_long <- tidyr::pivot_longer(
    model_metrics,
    cols = c("RMSE", "MAE", "R_Squared"),
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
# 12.8) XGBoost Parameter Tuning
#------------------------------------------------------------
# Define parameter grid
param_grid <- expand.grid(
  eta = c(0.01, 0.05, 0.1),         # Learning rate
  max_depth = c(4, 6),              # Maximum tree depth
  min_child_weight = c(1, 3, 5),    # Minimum sum of instance weights
  subsample = 0.8,                  # Subsample ratio (fixed)
  colsample_bytree = 0.8,           # Feature sampling ratio (fixed)
  gamma = 0.1                        # Minimum loss reduction (fixed)
)

print("Parameter tuning grid:")
print(param_grid)

# Initialize variables to store best parameters
best_params <- NULL
best_score <- Inf  # Lower RMSE is better
best_nrounds <- 0
cv_results_list <- list()

# Iterate through parameter grid
for (i in 1:nrow(param_grid)) {
  # Get current parameter combination
  current_params <- list(
    objective = "reg:squarederror",
    eval_metric = "rmse",
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
  
  # Get best iteration and score
  best_iter <- which.min(cv_result$evaluation_log$test_rmse_mean)
  current_score <- cv_result$evaluation_log$test_rmse_mean[best_iter]
  
  # Print current results
  cat("  Best iteration:", best_iter, "\n")
  cat("  Cross-validation score (RMSE):", current_score, "\n\n")
  
  # Update best parameters
  if (current_score < best_score) {
    best_score <- current_score
    best_params <- current_params
    best_nrounds <- best_iter
  }
}

# Print best parameters
print("Best parameter combination:")
print(best_params)
print(paste("Best number of rounds:", best_nrounds))
print(paste("Best score (RMSE):", best_score))

# Train final model with best parameters
set.seed(789)
xgb_model_tuned <- xgb.train(
  params = best_params,
  data = dtrain,
  nrounds = best_nrounds,
  watchlist = list(train = dtrain, test = dtest),
  verbose = 0
)

# Prediction
xgb_preds_tuned <- predict(xgb_model_tuned, dtest)

# Compute evaluation metrics after tuning
xgb_mse_tuned <- mean((test_data$disc_total - xgb_preds_tuned)^2)
xgb_rmse_tuned <- sqrt(xgb_mse_tuned)
xgb_mae_tuned <- mean(abs(test_data$disc_total - xgb_preds_tuned))
xgb_r2_tuned <- 1 - sum((test_data$disc_total - xgb_preds_tuned)^2) / 
  sum((test_data$disc_total - mean(test_data$disc_total))^2)

print("Performance of the Tuned XGBoost Model:")
print(paste("RMSE:", round(xgb_rmse_tuned, 4)))
print(paste("MAE:", round(xgb_mae_tuned, 4)))
print(paste("R²:", round(xgb_r2_tuned, 4)))

# Compute performance improvement
print("Performance Improvement:")
print(paste("RMSE Improvement:",
            round((xgb_rmse - xgb_rmse_tuned) / xgb_rmse * 100, 2),
            "%"))
print(paste("MAE Improvement:",
            round((xgb_mae - xgb_mae_tuned) / xgb_mae * 100, 2),
            "%"))
print(paste("R² Improvement:",
            round((xgb_r2_tuned - xgb_r2) / xgb_r2 * 100, 2),
            "%"))

# Get feature importance of the tuned model
xgb_importance_tuned <- xgb.importance(feature_names = colnames(x_train_xgb), model = xgb_model_tuned)
xgb_importance_df_tuned <- as.data.frame(xgb_importance_tuned) |>
  arrange(desc(Gain))

# Print feature importance
print("Feature Importance of the Tuned XGBoost Model (Top 20):")
print(head(xgb_importance_df_tuned, 20))

#------------------------------------------------------------
# 12.9) Findings and Predictions
#------------------------------------------------------------
## (1) Prediction of violence level at different genders


# Load necessary libraries
library(dplyr)
library(ggplot2)

# Get all possible values of gender
gender_levels <- levels(train_data$gender)
print("Gender categories:")
print(gender_levels)

# Create a dataframe to store prediction results for different gender categories
results <- data.frame(
  Gender = character(),
  DiscTotal_Prediction = numeric(),
  stringsAsFactors = FALSE
)

# Generate predictions for each gender category
for (gender_category in gender_levels) {
  # Randomly select samples from training data where gender equals the current category
  category_samples <- test_data %>%
    filter(gender == gender_category) %>%
    sample_n(min(50, sum(test_data$gender == gender_category)))
  
  if (nrow(category_samples) == 0) {
    next  # Skip categories with no samples
  }
  
  # Convert these samples to the format required by the model
  category_matrix <- model.matrix(~ . - disc_total, data = category_samples)[, -1]
  
  # Create DMatrix and make predictions
  dpredict <- xgb.DMatrix(data = category_matrix)
  disc_preds <- predict(xgb_model_tuned, dpredict)
  
  # Calculate average prediction
  avg_disc_pred <- mean(disc_preds)
  
  # Add to results
  results <- rbind(results, data.frame(
    Gender = gender_category,
    DiscTotal_Prediction = avg_disc_pred
  ))
}

# Sort results by prediction value for better visualization
results <- results %>% arrange(desc(DiscTotal_Prediction))

# Create factor with ordered levels based on prediction value
results$Gender <- factor(results$Gender, levels = results$Gender)

# Create bar chart of absolute predictions
p_absolute <- ggplot(results, aes(x = Gender, y = DiscTotal_Prediction, fill = DiscTotal_Prediction)) +
  geom_col() +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  labs(
    title = "Predicted Discrimination Total by Gender Category",
    x = "Gender",
    y = "Predicted Discrimination Total"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14),
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "none"
  )

# Print absolute prediction chart
print(p_absolute)

# Save chart
ggsave("gender_disc_total_absolute.png", p_absolute, width = 10, height = 6, dpi = 300)

## (2) Prediction of violence level at different minority status

# Get all possible values of minority status
minority_levels <- levels(test_data$minority)
print("Minority categories:")
print(minority_levels)

# Create a dataframe to store prediction results for different minority categories
results <- data.frame(
  Minority = character(),
  DiscTotal_Prediction = numeric(),
  stringsAsFactors = FALSE
)

# Generate predictions for each minority category
for (minority_category in minority_levels) {
  # Randomly select samples from test data where minority equals the current category
  category_samples <- test_data %>%
    filter(minority == minority_category) %>%
    sample_n(min(50, sum(test_data$minority == minority_category)))
  
  if (nrow(category_samples) == 0) {
    next  # Skip categories with no samples
  }
  
  # Convert these samples to the format required by the model
  category_matrix <- model.matrix(~ . - disc_total, data = category_samples)[, -1]
  
  # Create DMatrix and make predictions
  dpredict <- xgb.DMatrix(data = category_matrix)
  disc_preds <- predict(xgb_model_tuned, dpredict)
  
  # Calculate average prediction
  avg_disc_pred <- mean(disc_preds)
  
  # Add to results
  results <- rbind(results, data.frame(
    Minority = minority_category,
    DiscTotal_Prediction = avg_disc_pred
  ))
}

# Sort results by prediction value for better visualization
results <- results %>% arrange(desc(DiscTotal_Prediction))

# Create factor with ordered levels based on prediction value
results$Minority <- factor(results$Minority, levels = results$Minority)

# Create bar chart of predictions
p_absolute <- ggplot(results, aes(x = Minority, y = DiscTotal_Prediction, fill = DiscTotal_Prediction)) +
  geom_col() +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  labs(
    title = "Predicted Discrimination Total by Minority Category",
    x = "Minority Category",
    y = "Predicted Discrimination Total"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14),
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "none"
  )

# Print absolute prediction chart
print(p_absolute)

# Save chart
ggsave("minority_disc_total_prediction.png", p_absolute, width = 10, height = 6, dpi = 300)

# Print the results
print("Results sorted by Predicted Discrimination Total:")
print(results)




## (3) Prediction of violence level at different economic levels

# Get all possible values of economic level
econ_levels <- levels(test_data$econ_level)

# Create a dataframe to store prediction results for different economic levels
results <- data.frame(
  EconomicLevel = character(),
  DiscTotal_Prediction = numeric(),
  stringsAsFactors = FALSE
)

# Generate predictions for each economic level
for (econ_category in econ_levels) {
  # Randomly select samples from test data where econ_level equals the current category
  category_samples <- test_data %>%
    filter(econ_level == econ_category) %>%
    sample_n(min(50, sum(test_data$econ_level == econ_category)))
  
  if (nrow(category_samples) == 0) {
    next  # Skip categories with no samples
  }
  
  # Convert these samples to the format required by the model
  category_matrix <- model.matrix(~ . - disc_total, data = category_samples)[, -1]
  
  # Create DMatrix and make predictions
  dpredict <- xgb.DMatrix(data = category_matrix)
  disc_preds <- predict(xgb_model_tuned, dpredict)
  
  # Calculate average prediction
  avg_disc_pred <- mean(disc_preds)
  
  # Add to results
  results <- rbind(results, data.frame(
    EconomicLevel = econ_category,
    DiscTotal_Prediction = avg_disc_pred
  ))
}

# Keep the original ordered levels of economic status (from most difficult to easiest)
results$EconomicLevel <- factor(results$EconomicLevel, levels = econ_levels)

# Create a line chart to visualize the trend across economic levels
p_line <- ggplot(results, aes(x = EconomicLevel, y = DiscTotal_Prediction, group = 1)) +
  geom_line(size = 1.5, color = "blue") +
  geom_point(size = 4, color = "darkblue") +
  labs(
    title = "Trend of Predicted Discrimination Total Across Economic Levels",
    x = "Economic Level",
    y = "Predicted Discrimination Total"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

# Print line chart
print(p_line)

# Save line chart
ggsave("econ_level_disc_total_trend.png", p_line, width = 10, height = 6, dpi = 300)
