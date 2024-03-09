# Philadelphia Housing Price Predictor 

### Files 
- **Preprocessing and Models:** models.ipynb
- **Raw Data:** opa_properties_public.csv 
- **Model Scores:** model_scores.csv 

Various Models that predict the sale price of a home in Philadelphia. The following models are used in this project: 
- Linear Regression
- Decision Trees
- Random Forest
- XGBoost (Gradient Boosting) 

This project uses various machine learning models to try and predict the market value of a home in philadelphia. We begin by collecting a large dataset from the city of philadelphia (https://property.phila.gov/). The data is then cleaned, scaled, and encoded for proper model insertion. Statistics and correlation values are derived from the data to futhrer prune it and to check for multicolinearity among the features. 

# Data Cleaning & Preparation 
After loading in the raw CSV data, we obtained some statistics about the features. Most importantly, I needed to find the amount of missing data from each attribute and the particular data types associated with each attribute. On first observation, it was clear that the data set did not follow consistent labeling practices and data entry formats. 

### Dropping Useless Features 
features were dropped because they were redundant, indecipherable or irrelevant

### Features with Large Amounts of Missing Data 
I set a hard limit of 5% for missing values for a given feature. I then queried the dataset for features that exceeded this amount. Below is a graph oredered from most missing data to least showing this: 
![image](https://github.com/afeins19/PhillyHousingPricePredictor/assets/56986596/beba2a80-bb04-44c6-adf1-a9b95c0aabe7)

After looking at this data I elected to drop the following features 
```python 
attrs_exceed_missing = ['utility', 'unfinished', 'suffix',
'garage_type', 'number_of_rooms', 'separate_utilities',
'fuel', 'basements', 'central_air',
'type_heater', 'basements', 'building_code_description_new']
```

### Correlation with Target Variable 
I visualize some of the attributes with graphs and box-plots. This is done to determine the correlation between certain values for a given feature to our target (market value). This gives a good preliminary feel of the predictive power of our features. Below is one of these plots comparing the `exterior_condition` attribute to the `market_value` attribute. 

![image](https://github.com/afeins19/PhillyHousingPricePredictor/assets/56986596/fb298386-0d70-46d8-985c-342fd69aec63)

### NaN & Placeholder Values 
some of the data had values such as 1 or 0 for features like total_area which likely meant it was a placeholder and should be treated as a non-exitent value. The data was analyzed using datawrangler (a vscode plugin from Microsoft) to see which features had these placeholder values. They were then converted to proper NaNs in the dataset. 


# Feature Engineering 
the following feature engineering techniques were applied: 
- Imputation
- Encoding
- Scaling
- Outlier Prunning

### Imputation 
features which were deemed valuable and had relatively low amounts of missing data were imputed. The imputed features are listed here: 
```python
imputed_features = ['number_of_bathrooms', 'number_of_bedrooms', 
'number_stories', 'interior_condition', 'exterior_condition' , 
'year_built', 'total_livable_area', 'total_area', 'frontage', 
'taxable_building', 'exempt_building', 'depth', 'garage_spaces']
```

each feature was replaced by its repsective median value in the dataset. The reason for doing so, especially for features such as `quality_grade` was due to the fact these features were overwhelmingly cetnered around a specific value. Below is a graph showing the quality grade given to a home. On the x-axis is the grading scale (on a standard academic grading scale A+, A, A-, etc.) and with the number of values for each grade on the y-axis. 

![image](https://github.com/afeins19/PhillyHousingPricePredictor/assets/56986596/bf1f4092-39b8-42a4-a0ce-3d2ba2449e47)

### Encdoing 
Some of the data in this dataset was categorical, I apply the following encoding techniques to correctly format this data: 
- one-hot encoding
- binary encoding
- ordinal encoding

### Outlier Prunning 
using Datawrangler and an IQR (Inter-Quartile Range) analysis, I set hard values for the minimum and maximum of values of our features. Rows with features that did not fall within these limits were dropped so as to not skew the models. 

### Correlation Analysis 
A correlation matrix was created to show how each remaining correlated with the target variable (market value), below is the correlation data for various attributes. Not that categorically encoded attributes were suffixed with a numerical value such as `year_built_0`. 
```
Correlation Values for Each Attribute to the Target:

market_value                    1.000000
taxable_land                    0.978607
taxable_building                0.763275
total_livable_area              0.564986
frontage                        0.318940
total_area                      0.291106
number_stories                  0.254154
fireplaces                      0.233295
year_built_0                    0.226933
year_built_1                    0.214178
exempt_building                 0.208017
homestead_exemption             0.189623
general_construction_C          0.187049
garage_spaces                   0.184453
number_of_bathrooms             0.176232
year_built_2                    0.165221
depth                           0.156924
zip_code_0                      0.150231
...

```

Attributes with low correlations to the target attribute (which i defined as those with less than 15% correlation) were dropped. For future analysis and as a potential hyperparemeter, I would have considered adjusting this value and seeing how the models performed for different cut-offs. 

### Multicolinearity 
after performing some analysis, it was found that some features were colinear with each other. This didnt necessaly affect the accuracy of the models but made it difficult to determine which features specifically contributed to the predictive power of the models. I used Princple Component Analysis on the dataset for all models to effictively remove the problem of multicolinearity, this will be compared to models run on non-pca data. Below is a heatmap with each sqaure showing the relationship between the feature in the row of that sqaure and its colummn. 

![image](https://github.com/afeins19/PhillyHousingPricePredictor/assets/56986596/2c3e801b-8527-48f0-95b0-d868d328c619)


### Scaling 
All numerical attributes in the final feature set were scaled using standardization. 

$$
Z = \frac{X - \mu}{\sigma}
$$

we use the StandardScalar() from Scikit-learn which centers each feature about the mean and scales it by the standard deviation. This prevents features with higher magnitude values from dominating features with lower ranged values. Essentially, scaling in this way prevents the model from considering the magnitude of each feature directlly and more so where it lies within the mean of that feature. The data split into 2 new dataframes `data_numerical` and `data_categorical` which contain numerical and categorical values respectively. The numerical dataframe is then scaled and joined back in to the data. This will serve as the last feature engineering step before constructing the models. 

# Building the Models 
I employ the following models to make predictions on the market value of a given house: 
- Linear Regression
- Decision Trees
- Random Forest
- XGBoost (Gradient Boosting)

Before this is done, the data is split into 2 sets. Each Model is above is built twice, it is trained on the original features (split into test and train sets) as well as a test/train set with PCA applied to it. I created another hyperparameter `VARIANCE_PCA` which I set with an initial value of 0.95 (this is the value that the models will be trianed on for now). 

# Model Performance Evaluation  
After building and training the models, I placed the performance metrics into the graphs below. The top graph shows the differences in R^2 scores for all models with both PCA and Non-PCA feature sets. The lower graph shows the same but for RMSE (Root Mean Squared Error) - the loss function employed in the models. 

![image](https://github.com/afeins19/PhillyHousingPricePredictor/assets/56986596/c7220ef4-08be-42e9-8c2a-03d22babdb9b)

![image](https://github.com/afeins19/PhillyHousingPricePredictor/assets/56986596/a8aa8470-b800-4de1-9735-d686a43641fa)

# Discussion 

Training the models was a relatively straight forward process. It was interesting to not that our most effective model - Random Forest took roughly 10 minutes to train! This is likely due to the large quantity of rows (roughly 400,000) as well as about 20 columns. The large degree of branching required to build this model was substantial. It was unfortunate that a PCA resulted in roughly double the error across the board. In the future, I would have liked to tune which features PCA had selected and played with the `VARIANCE_PCA` hyperparameter to try and improve thse values a bit.  For now, lets just consider the non-pca models, since each model had an r^2 within 5% or even more, it is reasonable to go ahead and compare the RMSEs of these values since we know these models have high explanatory power over the data. I tabulated the performance of each model below: 


| Model                     | R-squared Train | R-squared Test | RMSE Train       | RMSE Test        |
|---------------------------|-----------------|----------------|------------------|------------------|
| Linear Regression         | 0.960352282     | 0.9620124098   | 27590.0816264309 | 27091.4913193792 |
| Linear Regression with PCA| 0.8781704898    | 0.8797881773   | 48324.4989806637 | 48367.4578600186 |
| Random Forest             | 0.9986926693    | 0.9916868227   | 5009.9759650896  | 12456.7796662126 |
| Random Forest with PCA    | 0.9932200283    | 0.9541479923   | 11391.5950350925 | 29828.8113418557 |
| XGBoost                   | 0.994284994     | 0.9882463032   | 10470.8033770846 | 15069.4710679681 |
| XGBoost with PCA          | 0.9466930938    | 0.934866723    | 31965.6152721463 | 35615.8873654423 |

