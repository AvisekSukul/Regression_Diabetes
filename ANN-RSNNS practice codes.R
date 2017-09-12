lapply(c("nnet",'neuralnet',"RSNNS"),function(x)require(x,character.only = T))
library(caret)
cntrl = trainControl("cv",number = 10)

# Regression Example
parkinson_data = read.csv("http://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data")
summary(parkinson_data)
parkinson_data$subject. = as.factor(parkinson_data$subject.)
parkinson_data$sex = as.factor(parkinson_data$sex)

set.seed(1234);test_index = sample(nrow(parkinson_data),0.2*nrow(parkinson_data))

train_data = parkinson_data[-test_index,]
test_data = parkinson_data[test_index,]

# Regression OLS
# ==============
model_lm = lm(motor_UPDRS~.,data = train_data[,-c(1,6)])
summary(model_lm)
library(MASS)
final_model = stepAIC(model_lm)
summary(final_model)
plot(final_model)

pred_ml = predict(final_model,newdata = test_data)
fscaret::RMSE(pred_ml,test_data$motor_UPDRS,1175)

# Using 10 fold cross validation
# ==============================
library(caret)
model_lm_cv = train(motor_UPDRS~.,data = train_data[,-c(1,6)],method = "lm", trControl = cntrl)
pred_ml_cv = predict(model_lm_cv,newdata = test_data)
fscaret::RMSE(pred_ml_cv,test_data$motor_UPDRS,1175)

# Regression ANN(nnet package)
# ============================
model_nnet = train(motor_UPDRS~.,data = train_data[,-c(1,6)],
                   method = "nnet", trControl = cntrl)
pred_nnet = predict(model_nnet,newdata = test_data)
fscaret::RMSE(pred_nnet,test_data$motor_UPDRS,1175)

# Regression using ANN after normalization (This process will take some time to 
# complete..Be patient..!!)
# ========================================
library(dummies)
train_data_dummy = dummy.data.frame(train_data[,-1])
test_data_dummy = dummy.data.frame(test_data[,-1])

train_data_dummy_norm = apply(train_data_dummy,2,
                              function(i)(i-min(i))/(max(i)-min(i)))
test_data_dummy_norm = apply(test_data_dummy,2,
                             function(i)(i-min(i))/(max(i)-min(i)))

tuneGrid = expand.grid(size = c(10,15,20),decay = c(0.02,0.05,0.1))

model_nnet_1 = train(y = train_data_dummy_norm[,5],x = train_data_dummy_norm[,-c(5,6)],
                   method = "nnet", trControl = cntrl,maxit = 10000,tuneGrid = tuneGrid)
pred_nnet_1 = predict(model_nnet_1,newdata = test_data_dummy_norm)

final_pred = min(train_data_dummy[,5]) + pred_nnet_1[,1]*(max(train_data_dummy[,5])-min(train_data_dummy[,5]))
fscaret::RMSE(final_pred,test_data_dummy[,5],1175)

# Regression RSNNS package
# ========================

model_mlp = mlp(x = train_data_dummy_norm[,-c(5,6)],y = train_data_dummy_norm[,5],size = c(5,4),
                maxit = 2000)

pred_mlp = predict(model_mlp,newdata = test_data_dummy_norm[,-c(4,5)])
final_pred_mlp = min(train_data_dummy[,5]) + pred_mlp[,1]*(max(train_data_dummy[,5])-min(train_data_dummy[,5]))
fscaret::RMSE(final_pred_mlp,test_data_dummy[,5],1175)

# Classification RSNNS package (simple example)
# ============================================
iris_dummy = dummy.data.frame(iris)
iris_dummy_norm = apply(iris_dummy,2,function(x)(x-min(x))/(max(x)-min(x)))
set.seed(12345);index = sample(150,120)
iris_train = iris_dummy_norm[index,]
iris_test = iris_dummy_norm[-index,]

class_model_mlp = mlp(x = iris_train[,c(1:4)],y = iris_train[,c(5:7)],size = c(2,3),maxit = 1000)
summary(class_model_mlp)
plotnet(class_model_mlp)
pred_model_mlp = predict(class_model_mlp,newdata = iris_test[,c(1:4)])
pred_model_mlp_final = apply(pred_model_mlp,1,function(i)levels(iris$Species)[which.max(i)])
caret::confusionMatrix(pred_model_mlp_final,iris$Species[-index])

# Classification RSNNS package
# ============================
bankloan = read.csv("~/Dropbox/Datasets/bankloan.csv",na.strings = "#NULL!")
bankloan_known = bankloan[1:700,]
bankloan_unknown = bankloan[701:850,]

bankloan_known$ed = as.factor(bankloan_known$ed)
bankloan_known$default = as.factor(bankloan_known$default)

bankloan_known_dummy = dummy.data.frame(bankloan_known)
set.seed(12345);index = sample(700,550)

bankloan_known_dummy_norm = apply(bankloan_known_dummy,2,function(x)(x-min(x))/(max(x)-min(x)))

bankloan_tarin = bankloan_known_dummy_norm[index,]
bankloan_test = bankloan_known_dummy_norm[-index,]

model_bankloan_mlp = mlp(x = bankloan_tarin[,c(1:12)],y = bankloan_tarin[,c(13,14)],size = c(5,3),maxit = 1000,learnFunc = "BackpropMomentum")

pred_bankloan_mlp = predict(model_bankloan_mlp,newdata = bankloan_test[,c(1:12)])
pred_bankloan_mlp_final = apply(pred_bankloan_mlp,1,function(i)levels(bankloan_known$default)[which.max(i)])
confusionMatrix(pred_bankloan_mlp_final,bankloan_known$default[-index])


