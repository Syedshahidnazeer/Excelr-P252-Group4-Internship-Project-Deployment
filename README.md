# Fake News Prediction

## Background
Social media and all it wonders has not only brought humanity together in a new and innovative way, but has also brought the rise of fake news for various purposes, from political to commercial and many other sectors. Fake news can misinform individuals and communities as a whole, persuading them to make decisions based on lies and deceit.

The spread of fake news is a major problem in society today and detecting such fake news is vital. How can we detect fake news you ask? With machine learning.



## Dataset
The dataset used to train our machine learning model was obtained from Kaggle, consisting of news articles and posted between 2015-2018.
Two sets of data were used, one consisting of all fake news, and another of all real news.
The overall format and columns were the same across both datasets.

The data consisted of the following fields:<br>
`title` `text` `subject` `date`

<a href="https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset">Link to dataset</a>



## Data Cleaning
The following data cleaning steps were taken:
1. Replace any empty strings with NaN
2. Add a binary column to each dataset, identifying the news articles as either being fake or real (0 = fake, 1 = real)
3. Append datasets into one dataframe
4. Create a new column called `combined_text` that concatenates the `title` and `text` columns
5. Pre-processing:
    - Cleaned text data from `combined_text` using regex (regular expressions)
    - Removed stop words (commonly used words in the English language, i.e. "a", "the", "is", "are")
    - Tokenized the text (splitting a sentence into individual words)
    - Lemmatized each word (removing inflections and grouping words by their lemma)



## Modeling
<p align="center">
  <img src="https://github.com/kevogil/fake-news-prediction/blob/4702295184e622e676b2e0690c3ab9e8146ffa6a/static/images/screenshots/ml_summary.png">
</p>

#### Logistic Regression
A logistic regression was the first model used, being a classification algorithm used to predict a discrete set of classes/categories (e.g., Yes/No, Young/Old, Happy/Sad). In our case, the discrete classes of fake and real news were mapped to a value of 0 and 1 respectively.

#### Decision Tree
A decision tree classification model was our second model used. The initial idea was that a decision tree might be a better fit for our model due to our data points not being easily separable. The flexibility of decision trees gave us our initial hypothesis that it would provide a higher accuracy score compared to a logistic regression model.

#### Multinomial Naive Bayes
A classification model that determines the probability of <b>A</b> happening, given that <b>B</b> has occurred.
The assumption is that the predictors/features are independent of one another, hence the term "naive".
The features/predictors used in this model was based off the frequency of words present in a given news article.
<br><br>
After testing all three models, it was clear that the decision tree classification model was more accurate, but only by a margin.
We were surprised at the accuracy of logistic regression, considering it being a more simpler model.
Ultimately, the slight increase in accuracy of decision trees did not justify the more complex modeling, and so we decided to stick with logistic regression for our machine learning model of choice.



## Visualizations

### Confusion Matrix
Logistic Regression was used to train our machine learning model.
The training data yielded in an accuracy score of 99.5%.
The testing data yielded in an accuracy score of 99.1%.
<p align="center">
  <img src="https://github.com/kevogil/fake-news-prediction/blob/720e122b2b890e01137c301a6a5413f69c2e6dd9/static/images/conf_matrix.png">
</p>

### Bar Chart
Our dataset could be classified into three main topics - Politics, World News, and U.S. News.
A majority of our training data consisted of news topics related to politics.
Understanding this, the confidence of the model returning an accurate prediction on non-political news is not great.
<p align="center">
  <img src="https://github.com/kevogil/fake-news-prediction/blob/720e122b2b890e01137c301a6a5413f69c2e6dd9/static/images/bar_chart.png">
</p>

### Stacked Bar Chart
We were interested in looking at the top words that showed up in our dataset, as well as what the distribution of the words being found in fake and real news looked like.
The top words were mainly related to U.S. presidents, most notably Donald Trump.
Interestingly, news articles related to Obama and Clinton had a higher distribution of fake news compared to Donald Trump.
<p align="center">
  <img src="https://github.com/kevogil/fake-news-prediction/blob/e1ac83ee95db5346d213fe21630fc985608df403/static/images/stacked_bar.png">
</p>

### Word Cloud
Our initial dataset consisted of 10,607,577 words across all of the news articles combined.
After pre-processing our data from removing stop words and lemmatization, our final dataset consisted of 214,805 unique words.
The following word cloud was a visual depiction of all of the unique words trained into our model, varying in size depending on the frequency they showed up across all trained news articles.
<p align="center">
  <img src="https://github.com/kevogil/fake-news-prediction/blob/720e122b2b890e01137c301a6a5413f69c2e6dd9/static/images/word_cloud.png", width="500", height="500">
</p>



## Identify Fake News
Try it yourself!<br>
Navigate to the <a href="https://3i2b9gpxzr.us-west-2.awsapprunner.com/form">Identify Fake and Real News</a> Page
<p align="center">
  <img src="https://github.com/kevogil/fake-news-prediction/blob/090396755f27c8e0cfc82412045fe015ac113d35/static/images/screenshots/form.png">
</p>
<br>
<p align="center">
    Find the latest politics news you've read or heard, paste it into the text box, and hit Verify!
</p>