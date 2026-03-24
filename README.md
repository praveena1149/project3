In this project we have to crate a job acceptance prediction application using Machine Learning.

1. First,we have to upload a dataset in a colab file.
2. Next, we have to handle the null values.
3. Next,we have to detect the outliers and we need to correct the outliers.
4. Then we have to check the inconsistencies for categorical features and then logical inconsistencies for all features.
5. Then we have to store the dataset in the database using sqlalchemy.
6. Next we have to create  a machine learning model for job acceptance prediction system.
7. In this first we have to do enconding for categorical features.
8. Next, we have to do scaling for numerical features.
9. Next we have to split the data set as X and y.
10. The target feature (status) is imbalanced.so we have to balance the target feature using Smote(oversampling) technique.
11. I choose RandomForest algorithm for this dataset,because i got more accuracy for this algorithm compared to other algorithmns.
12. next,we have to save a model using pickle library.
13. After Building a model,we have to do a streamlit application to predict whether the candidate is placed or not placed for new datapoints.
