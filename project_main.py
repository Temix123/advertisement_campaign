import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st 

st.markdown("""
# Advertisement Campaign

In this project I worked with an advertising data set, indicating whether or not a particular internet user clicked on an Advertisement on a company website. I created a model that will predict whether or not they will click on an ad based off the features of that user.

This data set contains the following features:

* 'Daily Time Spent on Site': consumer time on site in minutes
* 'Age': cutomer age in years
* 'Area Income': Avg. Income of geographical area of consumer
* 'Daily Internet Usage': Avg. minutes a day consumer is on the internet
* 'Ad Topic Line': Headline of the advertisement
* 'City': City of consumer
* 'Male': Whether or not consumer was male
* 'Country': Country of consumer
* 'Timestamp': Time at which consumer clicked on Ad or closed window
* 'Clicked on Ad': 0 or 1 indicated clicking on Ad
""")

ad_data = pd.read_csv('advertising.csv')

st.write(ad_data.head(5))

ad_data.info()

st.write(ad_data.describe())



st.markdown(""" ### AGE DISTRIBUTION IN THE DATASET""")
sns.set_style('whitegrid')

# Create the plot
fig, ax = plt.subplots()
sns.set_style('whitegrid')
ad_data['Age'].hist(bins=30)
ax.set_xlabel('Age')

# Display the plot
st.pyplot(fig)




# Create a jointplot
st.markdown(""" ### RELATIONSHIP BETWEEN AGE AND INCOME""")
joint_plot = sns.jointplot(x='Age', y='Area Income', data=ad_data)

# Render the jointplot in Streamlit
st.pyplot(joint_plot.figure)


st.markdown(""" ### RELATIONSHIP BETWEEN Clicked on Ad AND other features in the dataset""")
pairplot = sns.pairplot(ad_data,hue='Clicked on Ad',palette='bwr')
st.pyplot(pairplot.figure)



""" Logistic Regression

Now it's time to do a train test split, and train our model!

You'll have the freedom here to choose columns that you want to train on!
"""
from sklearn.model_selection import train_test_split

X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.linear_model import LogisticRegression


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)


with st.form("customer_input_form"):
    st.write("### Enter Customer Details")

    # Inputs
    time_spent = st.number_input(
        'Daily Time Spent on Site',
        min_value=ad_data['Daily Time Spent on Site'].min(),
        max_value=ad_data['Daily Time Spent on Site'].max(),
        step=0.1
    )

    Age = st.number_input(
        'Enter Age',
        min_value=ad_data['Age'].min(),
        max_value=ad_data['Age'].max(),
        step=1
    )

    Income = st.number_input(
        'Enter Area Income',
        min_value=ad_data['Area Income'].min(),
        max_value=ad_data['Area Income'].max(),
        step=10.0
    )

    Daily_internet_usage = st.number_input(
        'Enter Daily Internet Usage',
        min_value=ad_data['Daily Internet Usage'].min(),
        max_value=ad_data['Daily Internet Usage'].max(),
        step=1.0
    )

    Sex = st.radio(
        'Choose Sex', 
        options=["Male", "Female"]
    )

    # Submit Button
    submitted = st.form_submit_button("Submit")
sex2 = {"Male":1, "Female":0}
result = {1: "Click on Ad", 0:"Not Click on Ad"}
if submitted:
    main = ((logmodel.predict(np.array([time_spent,Age,Income,Daily_internet_usage, sex2[Sex]]).reshape(1,-1))[0])).astype("int")
    st.write(f" The Customer is likely to {result[main]}")
