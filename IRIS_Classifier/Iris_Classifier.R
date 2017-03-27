dataset <- read.csv(file="iris_sub_21655.csv", head= TRUE, sep =",")

#add new column called type and give it values 1 for setosa and -1 for versicolor
dataset$type[dataset$X %in% c("setosa")] <- 1
dataset$type[dataset$X %in% c("versicolor")] <- -1

#new variable vector for inputs and targets
sepal <- dataset[[2]]
petal <- dataset[[3]]
targets <- dataset[[5]]

#inputs matrix consist of 2 columns for sepal and petal lengths and 100 rows for 100 inputs vectors
inputs<-matrix(, nrow=100, ncol=2)
inputs[ ,1] <- sepal
inputs[ ,2] <- petal

#weights matrix 2*1 matrix having 2 weights generated randomly with a mean of 0 and standard deviation 0.1
weights<-matrix( rnorm(2*1,mean=0,sd=0.1), 2, 1)

#make sure all variables are matrices to apply matrix operations 
weights <- as.matrix(weights)
inputs <- as.matrix(inputs)
targets <- as.matrix(targets)

#activation matrix calculated for the firstly generated weights applied on the given inputs
activation <- inputs %*% weights
activation[activation > 0] <- 1
activation[activation <= 0] <- -1

#a clone for the activation matrix will be used for testing on new values
testActivation <- activation

#training algorithm takes the max number of iterations and the learning rate(eta) as parameter
train <- function(inputs, targets, maxIterations, eta) {
  i <- 0
  #keep training till the outputs are identical with the target or till we reach the max iterations 
  #to avoid looping infinetely
  while( i < maxIterations && all.equal(activation, targets) != TRUE) {
    #the error is the difference between the output(activation) and the target output
    error <<- activation - targets
    
    #plotting the error values and giving the output pdf a different name at each iteration
    pdfname <- paste(c("error", i), collapse = " ")
    pdfname <- paste(pdfname, ".pdf", sep = "")
    pdf(pdfname)
    plot(error)
    grid()
    
    
    dotproduct <- t(inputs) %*% error
  
    learningMatrix <- dotproduct * eta
    
    weights <<- weights - learningMatrix
    
    #recall using the new values calculated for the weights
    recall(inputs, weights)
    
    print(i)
    i <- i + 1
  }
  error <<- activation - targets
  
  pdf("finalerror.pdf")
  plot(error)
  grid()
  
}



#recall function calculates the new values for outputs(activation) using the newly calculated weights
recall <- function(inputs, weights) {
  activation <<- inputs %*% weights
  activation[activation > 0] <<- 1
  activation[activation <= 0] <<- -1
}

#test function similar to recall used to test new values on our trained model
test <- function(inputs, weights) {
  testActivation <<- inputs %*% weights
  testActivation[testActivation > 0] <<- 1
  testActivation[testActivation <= 0] <<- -1
}

train(inputs, targets, 10, 0.2)

testInputs = matrix( c(5.4, 6.3, 5.1, 1.7, 4.7, 3),  nrow=3,  ncol=2) 

test(testInputs, weights)


#ploting the petal lengths against the target outputs
pdf("petal.pdf")
plot(petal, targets)
grid()

cov(petal, targets)    #calculates covariance between petal length and targets
cor(petal, targets)   #calculates correlation between petal length and targets

#ploting the sepal lengths against the target outputs
pdf("sepal.pdf")
plot(sepal, targets)
grid()

cov(sepal, targets)    #calculates covariance between sepa length and targets
cor(sepal, targets)   #calculates correlation between sepal length and targets
