# Amazon-Review-Sentiment-Analysis
Applied multinomial naive bayes algorithm to predict sentiment based on text data. In this project we built model to perform sentiment analysis for alexa reviews on amazon to predict if customers are happy with the product or not

# 1. Understand The Problem Statement and Business Case

Natural language processing (NLP) can be used to build predictive models to perform sentiment analysis on social media posts and reviews and predict if customers are happy or not. NLP work by converting words into numbers and training a machine learning models to make predictions. That way, we can automatically know if our customers are happy or not without manually going through massive number of tweets or reviews.

In this project, we are going to use nlp for our predictive model to predict if customers are happy or not based on alexa reviews on amazon.

# 2. Import Libraries and Datasets

We used customer reviews dataset from kaggle that contains 3150 customer reviews on alexa product. The following is the first two rows of the dataset :

| rating  | date | variation | verified_reviews | feedback |
| ------------- | ------------- | ------------ | ------------- |------------- |
| 5 | 31-Jul-18 | Charcoal Fabric | Love my Echo! | 1 |
| 5  | 31-Jul-18 | Charcoal Fabric | Loved it! | 1 |
- rating: Rating of the products
- date : Date of the review
- variation : Variation of the products
- verified_reviews : Customers review
- feedback : boolean to say whether a customers is happy or not (1 = positive, 0 = negative)

# 3. Explore Dataset

## Checking missing values

Fortunately we don't have any missing values

## Data Visualization

![Data Vis 1](https://user-images.githubusercontent.com/107464383/196098503-7cf65ccb-6191-4548-b76c-6e30a4ed8e7f.PNG)

Majority of customers are happy with the products

![Data Vis 2](https://user-images.githubusercontent.com/107464383/196098624-adbc3729-e75c-42e3-87b3-d217afbfe5f2.PNG)

Majority of customers are giving 5 stars on product reviews

![Data Vis 3](https://user-images.githubusercontent.com/107464383/196098752-3dd27916-7bc1-4d85-a57f-a77cda472f66.PNG)

- Walnut Finish and Oak Finish is product variation with highest rating
- White is product variation with lowest rating

![Wordclout all](https://user-images.githubusercontent.com/107464383/196099347-ee94a484-f85a-47be-beff-9307874eb251.PNG)

The wordcloud above tell us the words that appear the most on reviews of the products

![Wordcloud Negative](https://user-images.githubusercontent.com/107464383/196099457-307bf4df-7443-48bd-92bc-fb28e120f424.PNG)

The wordcloud above tell us the words that appear the most on negative reviews of the products

# 4. Data Cleaning

## Drop Unnecesary Columns

We drop rating and date columns because we don't need them for our predictive model.

## Create Variation Dummies

We turn variation column into numerical data by creating dummies for variation column. The following is our variation dummies :

![Var Dummies](https://user-images.githubusercontent.com/107464383/196100743-c99af5da-566a-4ab9-ba78-d3d374aee49e.PNG)

The next thing to do is drop variation column on reviews dataset and concatinate it with our variation dummies.

# 5. Perform Data Cleaning by Applying Punctuation Removal, Stop Words Removal, and Count Vectorizer

We use nltk library to define a pipeline to clean up all the messages. The pipeline will peforms the following :
1. Remove punctuation (ex : , ! . ? / etc )
2. Remove stopwords ( ex : i, you, them , we, etc)

The following is our customer reviews after we apply our pipeline :

![Pipeline](https://user-images.githubusercontent.com/107464383/196119156-b03bc119-178d-47ef-86cc-4212b5e46a47.PNG)

Now we used count vectorizer to convert customer reviews into string data. The following is the result after we used count vectorizer to our data :

![Count Vectorizer](https://user-images.githubusercontent.com/107464383/196119729-0cfce777-0ff6-46b6-a4ab-10909698d759.PNG)

Finally we do the following before build the model :
1. Concat review dataset with vectorized reviews column
2. Drop verified_review and feedback columns

# 6. Train a Naive Bayes Classifier Model

We split the dataset into X Train, X Test, Y Train, and Y Test. We used 80% of our data into training dataset and 20% of our data into testing dataset. After split the dataset, we train our data using a MultinomialNB algorithm.

# 7. Asses Trained Model Performance

The main evaluation metric that we are used are confusion matrix and F1 score. The F-score, also called the F1-score, is a measure of a model's accuracy on a dataset. We can say F1-score is model accuracy. Confusion matrix is performance measurement for machine learning classification problem where output can be two or more classes. It is a table with 4 different combinations of predicted and actual values.

## Naive Bayes Classifier Model Confusion Matrix

![NB Cm](https://user-images.githubusercontent.com/107464383/196121222-1ff69d37-e720-40fa-87a5-a0ca4c035abe.PNG)

Based on matrix above, we correctly classify around 5.700 positives feedback and 17 negatives feedback. We misclassify 31 negatives feedback and 10 positives feedback.

## Naive Bayes Classifier Model F1 Score

![NB F1 Score](https://user-images.githubusercontent.com/107464383/196122894-1a50c563-3716-4469-8ddc-9026be737db5.PNG)

Table above shows that our logistic regression model have F1 Score of 0.93, it means accuracy of our Naive Bayes model is 93%

# 8. Train and Evaluate a Logistic Classifier Model

## Train a Logistic Classifier Model

We split the dataset into X Train, X Test, Y Train, and Y Test. We used 80% of our data into training dataset and 20% of our data into testing dataset. After split the dataset, we train our data using a Logistic Regression Classifier algorithm.

## Evaluate a Logistic Regression Model

### Logistic Regression Classifier Model Confusion Matrix

![Logistic Regression CM](https://user-images.githubusercontent.com/107464383/196125764-51041d6f-c65f-4ff4-a4c1-8fe51155ca3f.PNG)

Based on matrix above, we correctly classify around 5.800 positives feedback and 19 negatives feedback. We misclassify 4 negatives feedback and 29 positives feedback.

### Logistic Regression Classifier Model F1 Score

![Logistic Regression F1 Score](https://user-images.githubusercontent.com/107464383/196126152-9dc15fef-5577-4d1d-a729-04d9fccab31a.PNG)

Table above shows that our logistic regression model have F1 Score of 0.94, it means accuracy of our Naive Bayes model is 94%

# Conclusion

To predict if customers happy or not, we used two algorithms for our model. The following two algorithms are :

- Naive Bayes classifier with accuracy of 93%
- Logistic regression classifier with accuracy of 94%

We chose model with highest accuracy which is logistic regression with accuracy of 94%. Our model correctly predicted 5.800 positives feedback and 19 negatives feedback. Based on our model, we can tell that customers are quite happy with our products.















