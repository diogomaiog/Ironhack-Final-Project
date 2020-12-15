# Ironhack Data Analytics Final Project
[The working model](http://flask-env.eba-c8ddfpcy.eu-west-2.elasticbeanstalk.com/model)

**Author**: Diogo Maio Gonçalves 

**Contacts**: 
diogomaiog@gmail.com
[Linkedin](https://www.linkedin.com/in/diogo-m-goncalves/)

## Introduction
This project was made in one week, as a requirement for the completion of Ironhack's Data Analytics Bootcamp Program. 
I decided to make this project after reading about the concept [in an article](https://towardsdatascience.com/predicting-movie-profitability-and-risk-at-the-pre-production-phase-2288505b4aec). The author suggested doing NLP on the text description to improve the model's accuracy, that spiked my interest.

## Objective
The purpose of my solution is to **predict if a given movie idea is going to result in a profitable movie**.
The model atributes one out of three possible classifications to the idea:

1.  **Flop:** The movie has a ROI lower than -20%.
2.  **Regular:** The movies has a ROI between -20% and 100%
3.  **Blockbuster:** The movie has a ROI of over 100%

## Method

This model uses freely available [IMDb Datasets](https://www.imdb.com/interfaces/) complemented with additional information scrapped from the individual movie pages. It contains all the american made movies since 1980, that after all the necessary data cleaning total 7062 titles.

It uses a  [Naive Bayes Classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)  to predict the outcome based on the story text, and takes that as an input, along with the other variables shown in the form interface, to run an  [Extreme Gradient Boosting Classifier](https://en.wikipedia.org/wiki/Gradient_boosting)  to come up with the final prediction.

The created classifier has an **accuracy of 76.5%**  on test data.

To showcase the model, I **created an web application on AWS Elastic Beanstalk**.

## Included Files

The **Main.ipynb** file contains all the scripts necessary to generate the model and pickle files needed to run the **Prediction-Engine.ipynb**.
However, the [final scrapping script](https://colab.research.google.com/drive/1ZfGDsWBWuV3FsXZ1GTepGdZqZth1vznM?usp=sharing) wasn't run on Jupyter Notebook, but on Google Colab since it ran much faster there.
The contents of the /eb-Flask directory are the necessary files to deploy the AWS Elastic Beanstalk web application.
