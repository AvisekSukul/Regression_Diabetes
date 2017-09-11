library(FSelector)
f = paste("Hazard~",paste(names(train[,-c(1,2,3)]),collapse = "+"))
F = cfs(formula = as.formula(f),data = train)
new_formula = as.formula(paste("Hazard~",paste(F,collapse = "+")))
#without FS=========================================================================
train_new = train[,-c(1,2,3)]
test_new = test[,-c(1,2,3)]

library(caret)
dummies = dummyVars(~ ., data = train_new)
train_new = predict(dummies, newdata = train_new)
test_new = predict(dummies, newdata = test_new)

#Using Random Forest
library(randomForest)
set.seed(1000)
model1 = randomForest(train_new,train$Hazard, ntree=100, imp=TRUE, sampsize=10000, do.trace=TRUE)
pred_rf = predict(model1,test_new)
fscaret::RMSE(pred_rf,test$Hazard,10999)#3.79

#Using xgboost
paramList = list("objective" = "reg:linear", "nthread" = 8, "verbose"=0)

library(xgboost)
model2 = xgboost(param=param, data = train_new, label = train$Hazard, nrounds=2000, eta = .01, max_depth = 7,min_child_weight = 5, scale_pos_weight = 1.0, subsample=0.8) 
predict_xgboost=predict(model2, test_new)
fscaret::RMSE(predict_xgboost,test$Hazard,10999)#3.74

#with FS=============================================================================
train_new = train[,c(4,5,6,34,35)]
test_new = test[,c(4,5,6,34,35)]

library(caret)
dummies = dummyVars(~ ., data = train_new)
train_new = predict(dummies, newdata = train_new)
test_new = predict(dummies, newdata = test_new)

#Using Random Forest
library(randomForest)
set.seed(1000)
model1 = randomForest(train_new,train$Hazard, ntree=100, imp=TRUE, sampsize=10000, do.trace=TRUE)
pred_rf = predict(model1,test_new)
fscaret::RMSE(pred_rf,test$Hazard,10999)#3.93

#Using xgboost
paramList = list("objective" = "reg:linear", "nthread" = 8, "verbose"=0)

library(xgboost)
model2 = xgboost(param=param, data = train_new, label = train$Hazard, nrounds=2000, eta = .01, max_depth = 7,min_child_weight = 5, scale_pos_weight = 1.0, subsample=0.8) 
predict_xgboost=predict(model2, test_new)
fscaret::RMSE(predict_xgboost,test$Hazard,10999)#3.99
