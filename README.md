# STA_4103_Project1
> Authors: Bradley Genereux, Caleb Hancock, Kiersten Birkholz  
> Date: February 23, 2025

## **INSTRUCTIONS** ##
> 1) Clone the repository:
>    git clone https://github.com/BradleyGenereux/STA_4103_Project1.git
>    cd STA_4103_Project1
> 2) Set up the enviornment, ensure you have python installed by checking the version:
>    python --version
> 3) Install dependencies:
>    pip install -r requirements.txt
> 4) Run train-models-miami.py to create models
> 5) Run test_models.py to predict affordability based on user input

## **INTRODUCTION** ##
> This project explores how various factors in a dataset of Miami housing prices can help predict whether a person can afford a house. The dataset includes features such as property size (LND_SQFOOT, TOT_LVG_AREA), location (LATITUDE, LONGITUDE), proximity to important locations (e.g., RAIL_DIST, OCEAN_DIST, HWY_DIST), the age of the property (age), and its overall condition (structure_quality).

> The primary goal of this analysis is to determine the affordability of a house based on a person's budget and average income, and then predict whether that budget can accommodate houses with various parameters. By analyzing these factors, the project seeks to answer questions such as:

> 1) What parameters most significantly impact house pricing in Miami?
> 2) Can we predict if someone can afford a house based on their income, budget, and the features of available properties?

> The project uses machine learning models to estimate house prices and compare them to a person's budget, allowing for an exploration of which parameters make a house affordable or out of reach.

## **DOCUMENTATION** ##
> The dataset used in this project, Miami Housing Data, is publicly available. You can access the source documentation and more information about the data at the following URL:

> https://www.kaggle.com/datasets/deepcontractor/miami-housing-dataset

> This dataset contains detailed information about housing prices in Miami, as well as various features including their locations, sizes, and proximity to important landmarks and infrastructure. It was used to analyze the impact of these features on housing prices and to build a model that predicts house affordability.

## **ANALYSIS** ##
> The analysis in this project focuses on understanding the relationship between various house features and their sale prices, with the ultimate goal of predicting house affordability based on user-defined parameters. The following methods were used to analyze the dataset:

> Data Preprocessing and Feature Engineering:
> The dataset was cleaned and key features such as land square footage, total living area, and proximity to various locations (ocean, rail, highways) were selected. This step ensures that the features used in the model are relevant and prepared for analysis.

> Exploratory Data Analysis (EDA):
> Descriptive statistics were computed, and the relationships between features like house size and sale price were explored. Correlations were examined to identify the most significant factors influencing house prices, which include the total living area, proximity to important locations, and structure quality.

> Feature Selection and Model Choice:
> Features most strongly related to sale price were utilized in the predictive model. The dataset’s characteristics, including numerical variables, made linear regression a well-suited model for predicting house prices.

> Model Training and Prediction:
> A linear regression model was built to predict house prices based on features like size, age, and proximity to important locations. The model answers the question: "What type of house can I afford based on these parameters?"

> Model Evaluation:
> The model’s performance was evaluated using metrics such as Mean Squared Error (MSE) and R-squared. The predictions generated by the model were assessed to understand how accurately it predicts house affordability.

## **Conclusion** ##
> Our goal was to analyze the affordability of a house based on a person's budget and average income. Using various machine learning models, we were able to see how different parameters affected the prices, and to how extreme. 

> Despite the high accuracy of our models, they are not perfect. Outliers such as Luxury homes and properties in flood zone are some of the factors that can play a role in a decision tree being inaccurate which leads to the Random Forests also being inaccurate. Our models also don’t consider the fluctuations of the housing market. Shifts in real estate development, property taxes, natural disasters, are all smaller factors that can stack together to skew the modeling prediction.

> In our Analysis, we used logistic regressions, Support Vector Machines, Decision Trees, Random Forests, Naive Bayes models and K-Nearest neighbor to predict affordability of the houses in Miami. All the models performed well, however the best performing models were Decision Trees, Random Forests, and K-Nearest Neighbor. 

![Image](https://github.com/user-attachments/assets/baad4b3d-513c-4b20-b3d4-1bf6343d7a5c)

> Using Decision trees, we were able to leverage past house sales to make informed predictions on the affordability of houses with different parameters. Each “tree” considers a random subset of features (e.g., Square footage, Distance to the Railroad, Ocean, Highway). The Random Forest gives us the combined averages of each parameter giving strong conclusions about the affordability of a house on a given budget.

> The K-Nearest neighbor model found similarities between data points and compared the target home to similar houses recently sold in Miami. This model is adaptive to new data and is flexible and simple to use, helping homebuyers compare options based on past sales, and pairs well with real estate data. 

## **Future Enhancements** ##
> Although our mathematical models were highly accurate, there are many ways to improve the precision and accuracy to be to make an even better prediction on the affordability of houses in Miami. Hedonic Pricing Models are common in real estate economics and can incorporate inflation, interest rates, and market demand. Another possible tool we could use to improve our models is Time Series Forecasting. If we had the knowledge and resources to use this model it would give us data on the housing prices of 5-10 years so we could determine if the affordability of a house is attainable in a future point in time instead of limiting ourselves to just the present. 

> Another way we could improve our work is to incoporate more parameters to give homebuyers more options when selecting property. Although we had vital parameters like square footage, accessibilty, and age of the home, there are many paremeters that consumers would deem important in their homebuying journey. Crime rate, environmental risks, and distance to schools are just some of the parameters that could be included in the process of choosing a home in Miami.

## **CREDITS** ##
> Bradley Genereux - Coding  
> Caleb Hancock - README: Conclusion, Future Enhancements  
> Kiersten Birkholz - README: Instructions, Introduction, Documentation, Analysis
