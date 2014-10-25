In this project, the participants were asked to perform one set of 10
repetitions of the Unilateral Dumbbell Biceps Curl in five different
fashions according to certain specification Class A, Class B, Class C,
Class D and Class E. Class A represents the specific execution of the
exercise, while other classes represent the common mistakes. The data
for this project come from this source:
<http://groupware.les.inf.puc-rio.br/har>. The data was collected from
accelerometers on the belt, forearm, arm, and dumbell of six
participants.We are interested to predict how the Class 'classe
variable' depends on the other variables.In order to predict, first data
in "pml-training.cvs"was divided into training and testing data set. The
two models were developed based on regression tree and random forests
using training data set. The accuracy of models were cross-validated
with testing data set, which shows that the random forests model was
fitted better than regression tree and could predict the dependency of
classe variable on the other variables with high accuracy.

    library(knitr)
    library(caret)
    library(Hmisc)
    library(gridExtra)
    library(rpart)
    library(rpart.plot)
    library(rattle)
    library(randomForest)

DATA READING AND CLEANING

Read the training and testing files:

    read_trainingData <- read.csv("pml-training.csv")
    read_validatingData <- read.csv("pml-testing.csv")

Filtering training set:

1).Compare the variables in both training and testing data set provided
in .csv files.

2).Remove those variables from training data set, which contain missing
and empty observations in testing data set.

3).Save filtered training data set as sub\_read\_trainingData

    head(read_trainingData)
    head(read_validatingData)
    names <- colnames(read_trainingData)
    kurtosis_str <- grep("kurtosis_", colnames(read_trainingData)) 
    skewness_str <- grep("skewness_", colnames(read_trainingData))
    max_str <- grep("max_", colnames(read_trainingData))
    min_str <- grep("min_", colnames(read_trainingData))
    amplitude_str <- grep("amplitude_", colnames(read_trainingData))
    avg_str <- grep("avg_", colnames(read_trainingData))
    stddev_str <- grep("stddev_", colnames(read_trainingData))
    var_str <- grep("var_", colnames(read_trainingData))
    sub_read_trainingData <- subset(read_trainingData,select=-c(kurtosis_str,skewness_str,max_str,min_str,amplitude_str,avg_str,stddev_str,var_str))
    str(sub_read_trainingData)

Remove first seven columns from training data set since we are
interested to predict classe with belt,forearm,arm and dumbell only:

    sel_sub_read_trainingData <- sub_read_trainingData[,8:60]
    str(sel_sub_read_trainingData)
    head(sel_sub_read_trainingData)

Remove NA values:

    omit_sel_sub_read_trainingData <- na.omit(sel_sub_read_trainingData)
    str(omit_sel_sub_read_trainingData)

Creating training and testing sets for cross validation:

    inTrain <-createDataPartition(omit_sel_sub_read_trainingData$classe,p=0.7,list=FALSE) 
    fil_training <- omit_sel_sub_read_trainingData[inTrain,]
    fil_testing <- omit_sel_sub_read_trainingData[-inTrain,]

Some plots to explore

    featurePlot(x=fil_training[,c("roll_belt","roll_arm","roll_dumbbell","roll_forearm")],y=fil_training$classe,plot="pairs")

![](./MLproject_files/figure-markdown_strict/unnamed-chunk-7-1.png)

    # See Figure 1

    p1 <- qplot(roll_belt,roll_arm,colour=classe,data=fil_training)
    p2 <- qplot(roll_dumbbell,roll_forearm,colour=classe,data=fil_training)
    grid.arrange(p1,p2,ncol=2)
    p3 <- qplot(total_accel_belt,total_accel_arm,colour=classe,data=fil_training)
    p4<- qplot(total_accel_dumbbell,total_accel_forearm,colour=classe,data=fil_training)
    grid.arrange(p3,p4,ncol=2)

FITTING MODELS TO TRAINING DATA SET

1). First, tree model is fitted to fil\_training data set then the model
was evaluated using fil\_testing data set.

2). Second, random forest is fitted to fil\_training data set then the
model was evaluated using fil\_testing data set.

1). Fit a tree to training data set:

    set.seed(10000)  # For reproducibility
    # t_modFit <-train(classe~.,method="rpart",data=fil_training)    #too slow to work with large data set
    t_modFit <-rpart(classe~., data=fil_training)

i).Plots of tree

    plot(t_modFit)
    text(t_modFit,use.n=TRUE,all=TRUE, cex=0.6)
    # See Figure 2 

    prp(t_modFit) 

![](./MLproject_files/figure-markdown_strict/unnamed-chunk-11-1.png)

    #fancyRpartPlot(t_modFit,cex=0.5)

ii).Re-substitution error

    table(fil_training$classe, predict(t_modFit, type="class"))

iii).Predicting with test data set

    tree.pred <- predict(t_modFit,fil_testing,type="class")
    fil_testing$predRight <- tree.pred==fil_testing$classe
    p1t <- qplot(roll_belt,roll_arm,colour=predRight,data=fil_testing)
    p2t <- qplot(roll_dumbbell,roll_forearm,colour=predRight,data=fil_testing)
    grid.arrange(p1t,p2t,ncol=2,main="Test Data Predictions Using Tree Model")

![](./MLproject_files/figure-markdown_strict/unnamed-chunk-13-1.png)

    #See Figure 3

iv). Cross Validation:

a).The accuracy of model on testing data set

    AR_t <- sum(fil_testing$classe==tree.pred)/length(tree.pred)
    AR_t   

    ## [1] 0.7439252

AR\_t =0.743952

b).Out of sample error

    t_out_of_sample_error=1-AR_t 
    t_out_of_sample_error

    ## [1] 0.2560748

t\_out\_of\_sample\_error = 0.2560748

2). Fit a random forests:

    set.seed(10000)      # For reproducibility
    #rf_modFit <-train(classe~.,method="rf",data=fil_training); too slow to work with large data set
    rf_modFit <-randomForest(classe~., data=fil_training)
    rf_modFit

i).Plots of random forests

    varImpPlot(rf_modFit,main="Importance of random Forests", cex=.6)

![](./MLproject_files/figure-markdown_strict/unnamed-chunk-17-1.png)

    # See Figure 4

1.  Model evaluate:importance of variables produced by random forest
    (Gini Index)

<!-- -->

    gini <- importance(rf_modFit)
    df_gini <- as.data.frame(gini)
    order_gini <-gini[order(-df_gini$MeanDecreaseGini),]

ii).Re-substitution error

    table(fil_training$classe, predict(rf_modFit, type="class"))

iii).Predicting with test data set

    randomForest.pred <- predict(rf_modFit,fil_testing, type="class")

    fil_testing$predRight <- randomForest.pred==fil_testing$classe
    p1V <- qplot(roll_belt,roll_arm,colour=predRight,data=fil_testing)
    p2V <- qplot(roll_dumbbell,roll_forearm,colour=predRight,data=fil_testing)
    grid.arrange(p1V,p2V,ncol=2,main="New Data Predictions Using Random Forest Model")

![](./MLproject_files/figure-markdown_strict/unnamed-chunk-21-1.png)

    # See Figure 5

Confusion Matrix

    rf_CM <- confusionMatrix(fil_testing$classe,predict(rf_modFit,fil_testing))

iv). Cross Validation: a).The accuracy of model on testing data set

    AR_rf <- sum(fil_testing$classe==randomForest.pred)/length(randomForest.pred)         
    AR_rf  # This might not required since we already got accurancy from Confusion Matrix.

    ## [1] 0.9972812

AR\_rf=0.99728212

b).Out of sample error

    rf_out_of_sample_error=1-AR_rf 
    rf_out_of_sample_error

    ## [1] 0.002718777

rf\_out\_of\_sample\_error =0.00271877

The calculated out of samples errors are \~0.26 and 0.003 using
regression tree and random forests models. The figures for new data
predictions using random forest and regression tree models also show
that random forest makes very fewer mistake than regression tree model.

VALIDATING MODEL WITH TESTING DATA in "pml-testing.cvs"

Read the testing file

    read_validatingData <- read.csv("pml-testing.csv")

Cleaning up testing data as earlier :

    sub_read_validatingData <- subset(read_validatingData,select=-c(kurtosis_str,skewness_str,max_str,min_str,amplitude_str,avg_str,stddev_str,var_str))
    sel_sub_read_validatingData <- sub_read_validatingData[,8:60]
    omit_sel_sub_read_validatingData <- na.omit(sel_sub_read_validatingData)
    final_validatingdata <-omit_sel_sub_read_validatingData
    answers_rf <-predict(rf_modFit,final_validatingdata)
    answers_rf 

    ##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
    ##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
    ## Levels: A B C D E

SUBMISSION

    pml_write_files = function(x){
        n = length(x)
        for(i in 1:n){
            filename = paste0("./Answers/problem_id_",i,".txt")
            write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
        }
    }

    pml_write_files(answers_rf)

When 20 predications were submitted, all of the answers were passed.
