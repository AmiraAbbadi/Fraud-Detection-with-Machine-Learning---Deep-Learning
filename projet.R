# Charger les données
data <- read.csv("dataset_prepared.csv")

# Vérifier les premières lignes
head(data)

# Séparation des caractéristiques (X) et de la cible (y)
X <- data[, -which(names(data) == "Class")]
y <- data$Class

# Division des données en ensembles d'entraînement et de test
set.seed(42)
train_index <- createDataPartition(y, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Appliquer SMOTE si nécessaire (pour équilibrer les classes)
train_data_balanced <- SMOTE(Class ~ ., data = train_data, perc.over = 100, perc.under = 200)

# Entraînement du modèle de régression logistique
log_model <- glm(Class ~ ., data = train_data_balanced, family = binomial())
log_pred <- predict(log_model, test_data, type = "response")
log_pred_class <- ifelse(log_pred > 0.5, 1, 0)

# Confusion Matrix pour la régression logistique
log_conf_matrix <- confusionMatrix(as.factor(log_pred_class), as.factor(test_data$Class))
print(log_conf_matrix)

# Entraînement du modèle Random Forest
rf_model <- randomForest(Class ~ ., data = train_data_balanced, ntree = 100)
rf_pred <- predict(rf_model, test_data)

# Confusion Matrix pour Random Forest
rf_conf_matrix <- confusionMatrix(rf_pred, as.factor(test_data$Class))
print(rf_conf_matrix)

# Entraînement du modèle XGBoost
train_matrix <- as.matrix(train_data_balanced[, -ncol(train_data_balanced)])
train_label <- as.matrix(train_data_balanced$Class)
test_matrix <- as.matrix(test_data[, -ncol(test_data)])
test_label <- as.matrix(test_data$Class)

xgb_model <- xgboost(data = train_matrix, label = train_label, nrounds = 100, objective = "binary:logistic")
xgb_pred <- predict(xgb_model, test_matrix)
xgb_pred_class <- ifelse(xgb_pred > 0.5, 1, 0)

# Confusion Matrix pour XGBoost
xgb_conf_matrix <- confusionMatrix(as.factor(xgb_pred_class), as.factor(test_data$Class))
print(xgb_conf_matrix)
