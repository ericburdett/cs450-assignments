library(e1071)
library(datasets)
data(iris)

testSize = 0.3
gammaValue = 0.1
costValue = 100

# Iris Dataset
testRows <- sample(a:nrow(iris), trunc(nrow(iris) * testSize))
irisTest <- iris[testRows,]
irisTrain <- iris[-testRows,]

model <- svm(Species~., data = irisTrain, kernel = "radial", gamma = gammaValue, cost = costValue)
prediction <- predict(model, irisTest[,-ncol(iris)])

confusionMatrix <- table(pred = prediction, true = irisTest$Species)

correct <- prediction == irisTest$Species
accuracy <- prop.table(table(correct))

print("Iris Dataset")
#print(confusionMatrix)
print(accuracy)

# Letters Dataset
letters = read.csv(file = file.path("C:/School/CS450 - Machine Learning/cs450-assignments/R", "letters.csv"), header = T)

allRows <- 1:nrow(letters)
testRows <- sample(allRows, trunc(length(allRows) * testSize))

lettersTest <- letters[testRows,]
lettersTrain <- letters[-testRows,]

model <- svm(letter~., data = lettersTrain, kernel = "radial", gamma = gammaValue, cost = costValue)
prediction <- predict(model, lettersTest[,-1])

confusionMatrix <- table(pred = prediction, true = lettersTest$letter)

correct <- prediction == lettersTest$letter
accuracy <- prop.table(table(correct))

print("Letters Dataset")
#print(confusionMatrix)
print(accuracy)


# Vowels Dataset
vowels = read.csv(file = file.path("C:/School/CS450 - Machine Learning/cs450-assignments/R", "vowel.csv"), header = T)

allRows <- 1:nrow(vowels)
testRows <- sample(allRows, trunc(length(allRows) * testSize))

vowelsTest <- vowels[testRows,]
vowelsTrain <- vowels[-testRows,]

model <- svm(Class~., data = vowelsTrain, kernel = "radial", gamma = gammaValue, cost = costValue)
prediction <- predict(model, vowelsTest[,-13])

confusionMatrix <- table(pred = prediction, true = vowelsTest$Class)

correct <- prediction == vowelsTest$Class
accuracy <- prop.table(table(correct))

print("Vowels Dataset")
#print(confusionMatrix)
print(accuracy)
