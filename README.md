
---

<div align="center">    

# rainfall-prediction-using-BLSTM-GWO     
</div>

## Description   
This project code propose for my undergraduate thesis about "LSTM-BASED MODEL DEVELOPMENT WITH GRAY WOLF OPTIMIZER (GWO) ALGORITHM FOR RAINFALL PREDICTION".
As the name suggests, this project aims to predict rainfall in Banjarbaru.
There are 3 folders in this project.
1. Data Collecting
2. Model Development
3. Implementation App

Before exploring the whole of folders, you can install dependencies first
```bash
# Install dependencies
pip install -r requirements.txt
 ```   

## Data Collecting
In this folder, the data collection process is carried out using the Kaggle public dataset from the link (https://www.kaggle.com/datasets/greegtitan/indonesia-climate). The dataset is then selected to take the values in the city of Banjarbaru.
for more details, see the file data_collecting.ipynb

## Model Development
In this folder, the data that has been collected is preprocessed to match the existing model input format. There are 4 architectural models used in this study with optimization using the grey wolf organizer or not. Batch size and missing value handling were carried out to see the effect in this study.

## Implementation App
The models that have been built are compared and the best model is taken to be implemented into a rainfall prediction application. This application uses the Streamlit framework to run it.

## How to run
First, clone the project

```bash
# clone project   
git clone https://github.com/GerdaSofya/rainfall-prediction-using-BLSTM-GWO.git
 ```   

Next, navigate to any file and run it.   
 ```bash
# module folder
cd '.\Rainfall Prediction App\'
streamlit run .\rainfall_prediction.py
```
Enjoy the app