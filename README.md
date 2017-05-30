# useragentparser
Use random forest to parse user agent to extract browser family and major version

1.	The executable script train.py is written in python 3. Need package “httpagentparser”, “numpy”, “scipy” and “sklearn” installed to run it. 

2.	Run format is 
train.py --training <training_file> --test <test_file> --prediction <prediction_file>
You need provide the training file path, test file path and output prediction path to run the script. For example:
$python train.py --training data_coding_exercise.txt --test test_data_coding_exercise.txt --prediction predict.txt

3.	There are three functions in the script. The main() function take the optiaonal arguments, 
the parseUA() function take the input file as argument to parse the each user agent string into five dictionaries with keys in (‘os’, ‘platform’, ‘browser’, ‘dist’ and ‘bot’). 
The ‘bot’’s value is always False thus can be neglected. Take the other four keys as features with categorical values, also deal with the missing value. 
The report() function display the top parameter candidates from grid search.

4.	Use DictVectorizer class transform categorical features into vectors, and also store feature names. Then combine all vectors into training matrix. It has nearly 120 feature columns.

5.	Use LabelEncoder to encode user agent family label and take user agent version as categorical label.

6.	Split data into training data and cross validation data to train the model, and then check the bias and variance of the model.

7.	Use random forest classifier as base model for this problem due to its great performance and simplicity for classification. 
Due to time limit, haven’t tried SVM, gradient boosting tree or naïve Bayes classifier. Guess the performance should be the same or even worse.

8.	Use grid search to choose the best parameters for random forest. Due to time limit, tried very limit range of parameters. 

9.	Use optimal parameters calculate the accuracy for the whole training data set, the accuracy for user agent family is around 93.95% and for user agent version is around 94.65%.

10.	Apply model to test data. Then calculate test accuracy and write to output file with required format. 


Model: Random forest
Advantages: easy to implement, don’t need to tune key parameters like SVM, model is not sensitive to input parameters. 
Variance is significantly reduced by taking average of a set of decision trees.

Disadvantages: Time limit and computational resource limit restricts tuning the model to achieve better performance. 
Some issues exist for feature selecting, especially for the version parse in features. Script runs too slow, don’t have enough time to optimize the code. 
Only works for relative small dataset, should use Spark for real big data. 

