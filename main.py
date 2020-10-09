from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

st.image('https://res-1.cloudinary.com/crunchbase-production/image/upload/c_lpad,f_auto,q_auto:eco/ajracsdqu5gmyfl6nai0', width=200)
st.title('Lab Customer Analysis - Final Round')

st.header('01 - Problem (case study)')
st.subheader('Data Description')
st.subheader('Goal')
st.markdown('[Final round](https://github.com/ironhack-labs/lab-customer-analysis-final-round)',
            unsafe_allow_html=True)

st.header('02 - Getting Data')
github_url_csv_file = "https://raw.githubusercontent.com/ironhack-labs/lab-customer-analysis-final-round/master/files_for_lab/marketing_customer_analysis.csv"
st.markdown('[.csv file](https://raw.githubusercontent.com/ironhack-labs/lab-customer-analysis-final-round/master/files_for_lab/marketing_customer_analysis.csv)', unsafe_allow_html=True)

st.subheader('Raw Datas')
raw_file = pd.read_csv(github_url_csv_file)
st.write(raw_file)

clean_file = raw_file
st.header('03 - Cleaning/Wrangling/EDA')

st.subheader("Change headers names")
st.markdown('```clean_file.columns = clean_file.columns.str.lower()```')
clean_file.columns = clean_file.columns.str.lower()

st.subheader("Deal with NaN values")
clean_file.isnull().sum()
st.write(clean_file.isnull().sum())

st.subheader("Categorical Features")


st.subheader("Numerical Features")
st.subheader("Exploration")
st.markdown('```clean_file.describe()```')
st.write(clean_file.describe())

st.header("04 - Processing Data")
st.subheader("Dealing with outliers")

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

Q1 = clean_file.select_dtypes(include=numerics).quantile(0.05)
Q3 = clean_file.select_dtypes(include=numerics).quantile(0.95)

st.subheader('Factor')
factor = st.slider('Drop outliers by sliding the factor', 1, 10, 2)

file_num = clean_file.select_dtypes(include=numerics)[(clean_file.select_dtypes(
    include=numerics) > Q1) & (clean_file.select_dtypes(include=numerics) < Q3)]

st.write(clean_file)

st.subheader("Normalization")

st.subheader("Encoding Categorical Data")

file_cat = clean_file.select_dtypes(exclude=numerics)

ids = ["customer", "effective to date"]
# file_ids = file_cat(columns=ids)

file_cat = file_cat.drop(columns=ids)
file_cat = pd.get_dummies(file_cat)

encoder = OneHotEncoder(handle_unknown='error').fit(file_cat)
encoded = encoder.transform(file_cat).toarray()

st.write(file_cat)

# file = pd.concat([file_cat, file_num, file_ids])
file = pd.concat([file_cat, file_num])

st.subheader("Splitting into train set and test set")

targeted_value = "income"

X = file_num.drop([targeted_value], axis=1)
Y = file_num[targeted_value]

X = X.fillna(0)
Y = Y.fillna(0)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=15)

st.header("05 - Modeling")
st.subheader('Apply model')

model = linear_model.LinearRegression()

lm = model.fit(X_train, Y_train)
predictions = lm.predict(X_test)

lm.score(X_train, Y_train)
r2_score(Y_test, predictions)
st.write("The intercept is: ", lm.intercept_)
st.write("The slope is: ", lm.coef_)
mse = mean_squared_error(Y_test, predictions)
r2 = r2_score(Y_test, predictions)
st.write("Mean squared error", mse)
st.write("R2 score", r2)
