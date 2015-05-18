library(randomForest)
submit <- read.csv('sampleSubmission.csv')
test <- read.csv('test.csv')[,-1]
train <- read.csv('train.csv')[,-1]
model <- randomForest(train[, -ncol(train)], train$target)
submit[, 2:10] <- predict(model, newdata=test, type='prob')
write.csv(submit, 'submit.csv', quote=FALSE, row.names = FALSE)