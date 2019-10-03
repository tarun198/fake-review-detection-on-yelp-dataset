
Sellers selling products on the web often ask or take reviews from customers about the products that
they have purchased. The exponential growth e-commerce business has led to a tremendous increment
in the number of reviews provided. For a popular product, this number may go up to thousands or lacs.
For a genuine buyer, these reviews are of a lot of help. His decision of buying the product may depend
entirely upon these reviews provided by other customers. Therefore, user reviews play a legitimate role
in e-commerce platforms. However, this feature is often misused by sellers or companies to popularise
their products and make profits by generating a number of FAKE reviews that are intended to mislead
buyers into buying a particular product. This feature can also be misused to mislead the buyers by
generating FAKE negative reviews about a product of an opposition brand/company. Our project is
based on detection of the FAKE reviews.
We have used classification techniques like Support Vector Machine, Naïve Bayes, Decision Tree,
Linear Regression, etc. to analyse these fake reviews and predict the genuineness of the reviews. First,
we used Natural Language Processing techniques to “clean” the text and used this clean text as a
parameter along with parameters like Date of review, Reviewer ID and Product ID to train the dataset
based on above mentioned Classification Techniques and calculated their respective accuracies.


Follwings steps are implemented in order
1. First we imported all the essential libraries required namely pandas, numpy, sklearn,
matplotlib, etc.
2. Then we imported the dataset in the csv format which had 16002 reviews with equal number of
fake and genuine reviews.
3. Text wrangling and pre-processing of the Review Text in our dataset was the next step that we
implemented using NLP techniques:
3.1 Removing HTML text.
3.2 Removing accented characters
3.3 Expanding contractions of the text data of the review
3.4 Removing special characters
3.5 Stemming of the Review Text
3.6 Removing Stopwords
3.7 Creating Unigrams Matrix
3.8 Creating Bigrams Matrix
3.9 Percentage of POS Tagging
4. Adding features like Reviewer ID, Product ID, Date of review as independent variables along
with the cleaned review sparse matrix.
5. We even added the deviation of the customer’s rating from average product rating as an
independent variable.
6. Number of Positive and Negative Reviews has also been incorporated as an independent
variable.
7. Incorporating all the features into one dataset and choosing them randomly.
8. Creating Training set and Test Set by dividing the dataset into parts for training and prediction
purposes.
9. Applying Linear Discriminant Analysis to find a linear combination of features that
characterizes or separates the two classes of objects or events
10. Applying various classification models for prediction of fake and genuine reviews

10.1 Logistic Regression - Logistic regression predicts the probability of an outcome that can
only have two values (i.e. a dichotomy). The prediction is based on the use of one or
several predictors (numerical and categorical).



10.2 SVM - A Support Vector Machine (SVM) performs classification by finding the
hyperplane that maximizes the margin between the two classes. The vectors (cases) that
define the hyperplane are the support vectors.

10.3 Naïve Bayes - The Naive Bayesian classifier is based on Bayes’ theorem with the
independence assumptions between predictors. A Naive Bayesian model is easy to
build, with no complicated iterative parameter estimation which makes it particularly
useful for very large datasets.

10.4 Decision Tree - Decision tree builds classification or regression models in the form of a
tree structure. It breaks down a dataset into smaller and smaller subsets while at the
same time an associated decision tree is incrementally developed. The final result is a
tree with decision nodes and leaf nodes.

10.5 Random Forest - Random forest algorithm is a supervised classification algorithm. As
the name suggest, this algorithm creates the forest with a number of trees. In general,
the more trees in the forest the more robust the forest looks like. In the same way in the
random forest classifier, the higher the number of trees in the forest gives the high
accuracy results.

10.6 Kernel SVM - SVM algorithms use a set of mathematical functions that are defined as
the kernel. The function of kernel is to take data as input and transform it into the
required form. Different SVM algorithms use different types of kernel functions. These
functions can be different types. For example linear, nonlinear, polynomial, radial basis
function (RBF), and sigmoid.

11. Applying 5-fold cross validation technique after each classification model to avoid overall
over/under fitting on our dataset.
