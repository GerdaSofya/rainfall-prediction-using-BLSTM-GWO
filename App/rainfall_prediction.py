import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K


def rmsle(y_true, y_pred):
    return K.sqrt(K.mean(K.square(K.log(y_pred + 1) - K.log(y_true + 1))))


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def inverse_normalization(value, min_value, max_value):
    return value * (max_value - min_value) + min_value


def predict_rainfall(data_input):
    min_value = [0, 19, 23.8, 0, 1, 0, 0, 0]
    max_value = [213.9, 27.9, 36.8, 11, 13, 5, 360, 8]

    data_input_normalize = []
    for i in range(len(data_input)):
        normalize = (data_input[i] - min_value[i]) / \
            (max_value[i] - min_value[i])
        data_input_normalize.append(normalize)

    input_array = np.array(data_input_normalize)
    input_array = input_array.reshape((1, 1, input_array.shape[0]))

    model = load_model('model.h5', custom_objects={
                       'rmsle': rmsle, 'rmse': rmse})

    predict = model.predict(input_array)

    min_RR = 0
    max_RR = 213.9

    result = []
    for i in range(len(predict[0])):
        result.append(inverse_normalization(predict[0][i], min_RR, max_RR))

    return result


def main():
    st.title("Rainfall Prediction")
    st.write("This application functions to predict rainfall for the next 8 days using the Bidirectional LSTM model with the Gray Wolf Optimizer algorithm")

    st.header("Please enter the following climate parameters:")
    RR = st.number_input("1. Precipitation today:")
    Tn = st.number_input("2. Minimum temperature today:")
    Tx = st.number_input("3. Maximum temperature today:")
    ss = st.number_input("4. Long duration of sunshine today:")
    ff_x = st.number_input("5. Maximum wind speed today:")
    ff_avg = st.number_input("6. Average wind speed today:")
    ddd_x = st.number_input("7. Wind direction at today's maximum wind speed:")
    ddd_car = st.selectbox(
        '8. Most wind direction today:',
        ('C', 'E', 'N', 'NE', 'NW', 'S', 'SE', 'SW', 'W')
    )
    # Make Button
    clicked = st.button("Predict")

    # Checks if the button is pressed
    if clicked:
        ddd_car_code = {0: 'C', 1: 'E', 2: 'N', 3: 'NE',
                        4: 'NW', 5: 'S', 6: 'SE', 7: 'SW', 8: 'W'}
        ddd_car_value = list(ddd_car_code.keys())[
            list(ddd_car_code.values()).index(ddd_car)]

        data_input = [RR, Tn, Tx, ss, ff_x,
                      ff_avg, ddd_x, float(ddd_car_value)]

        result = predict_rainfall(data_input)

        series = {
            "Days to-": [i+1 for i in range(len(result))],
            "rainfall": result
        }

        st.header("Prediction result")
        st.line_chart(series["rainfall"])
        st.table(series)


if __name__ == "__main__":
    main()
