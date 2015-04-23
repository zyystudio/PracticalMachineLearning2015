#Load all necessary Libraries
library(caret)
library(rattle)
library(rpart)
library(randomForest)
library(e1071)

#Set seed for research reproducibility
set.seed(1234)

#Getting the train and test files URL:
trainUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
directory <- getwd()
download.file(trainUrl, paste(directory, "/pml-training.csv", sep=""), method="curl")
download.file(testUrl, paste(directory, "/pml-testing.csv", sep=""), method="curl")
training <- read.csv("pml-training.csv", header=TRUE, sep=",", na.strings=c("", "NA", "NULL"))
testing <- read.csv("pml-testing.csv", header=TRUE, sep=",", na.strings=c("", "NA", "NULL"))

dim(training)
dim(testing)
totalNAvaluesByColumnsInTrain <- colSums(!is.na(training))
totalNAvaluesByColumnsInTest <- colSums(!is.na(testing))


# If it has more than 40% of NA values, we will remove it.
goodTrainData <- totalNAvaluesByColumnsInTrain>0.6*(nrow(training))
goodTrainDataName <- names(goodTrainData[goodTrainData==TRUE])
goodTestData <- totalNAvaluesByColumnsInTest>0.6*(nrow(testing))
goodTestDataName <- names(goodTestData[goodTestData==TRUE])

# Keep only the good columns in both train and test sets.
training <- training[goodTrainData]
testing <- testing[goodTrainData]


# Partition data
inTrain <- createDataPartition(y=training$classe, p=0.6, list=FALSE)
myTraining <- training[inTrain,]
myTesting <- training[-inTrain,]


# removing all first column variable `x` as it cannot be used as a predictor.
myTraining <- subset(myTraining, select= -X)
myTesting <- subset(myTesting, select= -X)

#The out sample error rate from rpart model is: r 1-CMRPart$overall[1]. 
#This out sample error is quite high. We will not use this predictive model.

## Using ML algorithms for prediction: Random Forests
modFitRF <- randomForest(classe ~. , data=myTraining)
## Prediction on myTesting dataset
predictionsRF <- predict(modFitRF, myTesting, type = "class")
## ConfusionMatrix
CMRF <- confusionMatrix(predictionsRF, myTesting$classe)
# Accuracy
CMRF$overall[1]
CMRF[3]

# function for writing the output files
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
# write the predictons generated in the previsou step into files.
pml_write_files(predictionsRF)