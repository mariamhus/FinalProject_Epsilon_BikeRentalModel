import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Title
st.title('Count of regestered and casual Users for Bike Rental\n')



# Load Cleaned Data
df = pd.read_csv('cleaned_day')

# Load Preprocessor
preprocessor = pd.read_pickle('preprocessor.pkl')
model = pd.read_pickle('model.pkl')


# Input Data

season = st.selectbox('season', df['season'].unique())
Year = st.selectbox('Year', df['Year'].unique())
month = st.selectbox('month', df['month'].unique())
holiday = st.selectbox('holiday', df['holiday'].unique())
weekday = st.selectbox('weekday', df['weekday'].unique())
workingday = st.selectbox('workingday', df['workingday'].unique())
temp = st.number_input('temp', df.temp.min(), df.temp.max())
atemp = st.number_input('atemp', df.atemp.min(), df.atemp.max())
weathersit = st.selectbox('weathersit', df['weathersit'].unique())
humidity = st.number_input('humidity', df.humidity.min(), df.humidity.max())
windspeed = st.number_input('windspeed', df.windspeed.min(), df.windspeed.max())


# Preprocessing
new_data = {'season': season, 'Year': Year, 'month': month, 'holiday': holiday, 'weekday': weekday,
         'temp': temp, 'atemp': atemp, 'weathersit': weathersit, 'humidity': humidity,
         'windspeed':windspeed, 'workingday':workingday}
new_data = pd.DataFrame(new_data, index=[0])
new_data_preprocessed = preprocessor.transform(new_data)


# Prediction
log_count = model.predict(new_data_preprocessed) # in log scale

# new_per = np.round(log_count,1)
# new_per = str(log_count)[4:6]+"."+ str(log_count)[6:7]+"%"

# Output
if st.button('Predict'):
    st.markdown('# Count Of Bike Rentals Is:')
    st.markdown(log_count)
    # st.markdown(new_per)

