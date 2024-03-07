import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

# for converting figure to a image
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
import base64

#mean absolute scaled error(the smaller the better)
def mase(forecast, actual):
   
    forecast = np.array(forecast)
    actual = np.array(actual)
    
    # Calculate Mean Absolute Error (MAE)
    mae = np.mean(np.abs(forecast - actual))
    
    # Calculate mean absolute difference between consecutive actual values
    diff = np.diff(actual)
    mean_abs_diff = np.mean(np.abs(diff))
    
    # Calculate MASE
    mase_score = mae / mean_abs_diff
    
    return round(mase_score, 3)


def perform_analysis(deviceId_to_filiter, dataset_choice):
    dataset_choice = str(dataset_choice) + ".xlsx"
    df = pd.read_excel(dataset_choice)


    #get the column names
    month = str(df.columns[0])
    value = str(df.columns[1])
    device_id = str(df.columns[2])


    # if device id is specified
    if deviceId_to_filiter != "none":
        deviceId_to_filiter = int(deviceId_to_filiter)
        
        df = df[df[device_id] == deviceId_to_filiter]

        # reset the index of filiterd df
        df.reset_index(drop=True, inplace=True)

        # convert Month column to time object
        df[month] = pd.to_datetime(df[month])

        #if there ara missing months replace them with 0 value
        new_rows = []
        firstMonth = df.iloc[0][month]
        lastMonth = df.iloc[-1][month]

        currentMonth = firstMonth
        for index, row in df.iterrows():
            if row[month] == lastMonth:
                break
            while row[month] > currentMonth + pd.DateOffset(months=1):
                # Create a new row with currentMonth + 1 month value and column name value with 0
                new_row = {month: currentMonth + pd.DateOffset(months=1), value: 0, device_id: deviceId_to_filiter}
                new_rows.append(new_row)
                currentMonth += pd.DateOffset(months=1)
            
            currentMonth = row[month]

        # append the new rows to the original DataFrame
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

        # sort the DataFrame by month
        df.sort_values(month, inplace=True)

        # reset the index
        df.reset_index(drop=True, inplace=True)

        df.drop(device_id, axis=1, inplace=True)

        df= df.set_index([month])

        
        plt.figure(figsize=(10, 6))
        plt.plot(df.index, df[value], marker='o', linestyle='-')
        plt.title(f'{value} over Time for Device Id {deviceId_to_filiter}')
        plt.xlabel('Month')
        plt.ylabel('Consignment')
        plt.grid(True)

        # Convert the matplotlib figure to an image
        img_data = io.BytesIO()
        plt.savefig(img_data, format='png')
        img_data.seek(0)
        currentGraph = base64.b64encode(img_data.read()).decode('utf-8')

    # if device id is not specified
    else:
        # drop the device id column since it is not important
        df.drop(device_id, axis=1, inplace=True)

        # convert Month column to time object
        df[month] = pd.to_datetime(df[month])

        df = df.groupby(month)[value].sum().reset_index()

        df= df.set_index([month])

        plt.figure(figsize=(10, 6))
        plt.plot(df.index, df[value], marker='o', linestyle='-')
        plt.title(f'{value} over Time for All Devices')
        plt.xlabel(month)
        plt.ylabel(value)
        plt.grid(True)

        # Convert the Matplotlib figure to an image
        img_data = io.BytesIO()
        plt.savefig(img_data, format='png')
        img_data.seek(0)
        currentGraph = base64.b64encode(img_data.read()).decode('utf-8')

    
    #Model building
    try:
        # we find the best parameter values for the arime model ARIMA(p,d,q)
        stepwise_fit = auto_arima(df[value], trace=True)
    except:
        predictionGraph = None
        result_error = None
    else:
        #test train split
        X = df[value].values
        train_size = int(len(X) * 0.66)
        train, actual = X[:train_size], X[train_size:]

        history = train.tolist() 
        predictions = []

        for time_step in range(len(actual)):
            current_model = ARIMA(history, order=stepwise_fit.order)
            current_model_fit = current_model.fit()
            current_output = current_model_fit.forecast()
            predicted_value = current_output[0]
            predictions.append(predicted_value)
            true_value = actual[time_step]
            history.append(true_value)
            

        # Error rate for time series if it is smaller then 1 it is a good model
        result_error = mase(actual, predictions)
        

        finalModel = ARIMA(df, order = stepwise_fit.order).fit()

        # we can change the number of months we want to predict 
        number_of_months_to_predict = 2
        prediction = finalModel.predict(len(df), len(df) + number_of_months_to_predict)

        
        plt.figure()
        df.plot(legend=True, label="Train", figsize=(10, 6))
        prediction.plot(legend=True, label="Prediction")

        # Convert the Matplotlib figure to an image 
        last_graph_data = io.BytesIO()
        plt.savefig(last_graph_data, format='png')
        last_graph_data.seek(0)
        predictionGraph = base64.b64encode(last_graph_data.read()).decode('utf-8')

    return {
        'currentGraph': currentGraph,
        'error': result_error,
        'predictionGraph': predictionGraph
    }
