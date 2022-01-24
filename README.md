# Starbucks-Capstone-Challenge-Capstone Project for Udacity’s Data Scientist Nanodegree
Read more here --> https://medium.com/@kris.klimenchuk/starbucks-capstone-challenge-d9f274666199

### Project Intro
This data set contains simulated data that mimics customer behavior on the Starbucks rewards mobile app. Once every few days, Starbucks sends out an offer to users of the mobile app. An offer can be merely an advertisement for a drink or an actual offer such as a discount or BOGO (buy one get one free). Some users might not receive any offer during certain weeks. 
Not all users receive the same offer, and that is the challenge to solve with this data set. Task is to combine transaction, demographic and offer data to determine which demographic groups respond best to which offer type.

### Methods Used
- Pandas, Numpy for cleaning, manipulation and transform data
- Matplotlib and Seaborn for  visualization
- Scikit-learn: for building and evaluating classifier models


### Needs of this project:
•	Explore, validate and clean data

•	Merging three datasets into one

•	Exploratory Data Analysis

•	Pre-processing the data

•	Building Pipelines for each classifier: scaling the numerical features, using GridSearchCV to find the best parameters

•	Evaluating the models using the metrics: Accuracy, Recall, Precision, F1

### Files specifications
There the three data sets:
1.	Portfolio df: dataset describes each offer type's characteristics, including its offer type, difficulty, and duration.
2.	Profile df: dataset containing information about customer demographics: age, gender, income, and the date since they are members.
3.	Transcript df: information about when a customer made a purchase, viewed an offer, received an offer, and completed an offer. It's important to note that if a customer completed an offer but never actually viewed the offer, this does not count as a successful offer as the offer could not have changed the outcome.

The structure for each df:

1.1 Portfolio DataFrame
_portfolio.json_
- id (string) - offer id
- offer_type (string) - a type of offer: BOGO, discount, informational
-	difficulty (int) - the minimum required to spend to complete an offer
-	reward (int) - the reward is given for completing an offer
-	duration (int) - time for the offer to be open, in days
- channels (list of strings)

1.2 Profile DataFrame
_profile.json_
- age (int) - age of the customer
- became_member_on (int) - the date when the customer created an app account
- gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
- id (str) - customer-id
•	income (float) - customer's income

1.3 Transcript DataFrame
_transcript.json_
- event (str) - record description (transaction, offer received, offer viewed, etc.)
- person (str) - customer-id
- time (int) - time in hours since the start of the test. The data begins at time t=0
- value - (dict of strings) - either an offer id or transaction amount depending on the record

The main goal of the project builds a model for the classification of successful or unsuccessful offers. 
The business goal is to find: the most successful offer, the most profitable offer, customers who purchase by that offer.

### Finding the best model
I have prepared a Pipeline for each model it helps to find the best model. To evaluate models, I used the following metrics: accuracy, precision, recall, f1-score. 
The preferred models:
- Logistic Regression
- K-Nearest Neighbors
- Decision Tree Classifier
- Random Forest: ensemble bagging classifier

### Summary
- Cleaning the df to make sure it is no noise in the data.
- Combine all tree data sets into one. It was the tricky part; thanks to the Udacity mentors for helping here.
- Implemented EDAs on the datasets to determine how the different demographics of customers reacted to the various offer types. Finding the best offer in the data set.
- Preprocessed the data to ensure it was ready for the predictive algorithms.
- I used several models to predict which proposal would be successful.

**References:**

[https://towardsdatascience.com/dealing-with-list-values-in-pandas-dataframes-a177e534f173](https://towardsdatascience.com/dealing-with-list-values-in-pandas-dataframes-a177e534f173)

[https://www.geeksforgeeks.org/python-program-to-swap-keys-and-values-in-dictionary/](https://www.geeksforgeeks.org/python-program-to-swap-keys-and-values-in-dictionary/)

[https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea](https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea)

[https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)

[https://machinelearningmastery.com/precision-recall-and-f-measure-for-imbalanced-classification/](https://machinelearningmastery.com/precision-recall-and-f-measure-for-imbalanced-classification/)

https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9

[https://towardsdatascience.com/normalization-vs-standardization-explained-209e84d0f81e](https://towardsdatascience.com/normalization-vs-standardization-explained-209e84d0f81e)

[https://medium.com/swlh/quick-guide-to-labelling-data-for-common-seaborn-plots-736e10bf14a9](https://medium.com/swlh/quick-guide-to-labelling-data-for-common-seaborn-plots-736e10bf14a9)

[https://github.com/Surveshchauhan/StarbucksKNowledge](https://github.com/Surveshchauhan/StarbucksKNowledge)

[https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.loc.html](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.loc.html)

[https://pandas.pydata.org/docs/reference/api/pandas.concat.html](https://pandas.pydata.org/docs/reference/api/pandas.concat.html)

[https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

[https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm](https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm)

[https://scikit-learn.org/stable/modules/tree.html](https://scikit-learn.org/stable/modules/tree.html)


