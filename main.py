import os 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

def main():
    pd.options.mode.chained_assignment = None  # default='warn'

    '''

        Get full path of all data files in the pbp_data folder

    '''

    project_dir = "." # Uses the current directory
    szn_folders = os.listdir(f"{project_dir}/pbp_data") # Array of file names in pbp_data folder

    #print(pbp_data) # Prints out the files in the pbp_data directory

    target_seasons = [x for x in szn_folders 
                  if ('1999' in x) |
                  ('2000' in x) |
                  ('2001' in x) |
                  ('2002' in x) |
                  ('2003' in x) |
                  ('2004' in x) |
                  ('2005' in x) |
                  ('2006' in x) |
                  ('2007' in x) |
                  ('2008' in x) |
                  ('2009' in x) |
                  ('2010' in x) |
                  ('2011' in x) |
                  ('2012' in x) |
                  ('2014' in x) |
                  ('2013' in x) |
                  ('2014' in x) |
                  ('2015' in x) |
                  ('2016' in x) |
                  ('2017' in x) |
                  ('2018' in x) |
                  ('2019' in x) | 
                  ('2020' in x) | 
                  ('2021' in x)]

    data_files = ([f"""{project_dir}/pbp_data/{data_folder}/{os.listdir(f"{project_dir}/pbp_data/{data_folder}")[0]}""" for data_folder in target_seasons])
    #print(data_files) # Prints out the files in the pbp_data directory with full path

    '''

        Use Pandas to initialize the data frame with CSV data

    '''

    df = pd.DataFrame() # Create data frame variable

    # Loop through each file and add it to the data frame
    for file in data_files:
        df = pd.concat([df, pd.read_csv(file, low_memory = False)], ignore_index = True)

    df = df.reset_index(drop=True) # Reset row indexes so when multiple files are appended there is no overlap

    print(df.shape) # Print out the rows and columns of the data frame
    #print(df.sample(20)) # Print a random sample of 20 rows of data

    '''

        Get QB stats

    '''

    # Select the stat categories and sorting data
    qb_stats = ['season', 'passer_id', 'passer', 'pass', 'complete_pass', 'interception', 'sack', 'yards_gained', 'touchdown']
    group_by_stats = ['season', 'passer_id', 'passer']

    print(type(qb_stats))
    print(type("test"))

    qb_df = (df.loc[:, qb_stats].groupby(group_by_stats, as_index = False).sum()) # Group the data and aggregate the sum

    #print(qb_df.sample(20)) # Print a random sample of 20 rows of data

    print("Creating CSV file with QB stats by year...")

    overwrite_csv = ""
    csv_filename = f"{project_dir}/usable_data/qb/stats_by_year.csv"
    ask_to_overwrite_csv(overwrite_csv, qb_df, csv_filename)

    print("Creating PNG files graphing QB stats by touchdowns...")

    overwrite_png = ""
    png_path = f"{project_dir}/usable_data/qb/touchdowns_and_"
    png_x = 'touchdown'
    png_y = ['yards_gained', 'complete_pass', 'pass', 'interception','sack']
    ask_to_overwrite_graph(overwrite_png, qb_df, png_path, png_x, png_y)

    '''

        Create futures data

    '''

    _df = qb_df.copy() # Create a copy of the data frame

    _df['season'] = _df['season'].add(1) # Add one to the year

    # Merge the two datasets using a left join to add the 'previous' season data
    new_qb_df = (qb_df.merge(_df, on=['season', 'passer_id', 'passer'], suffixes = ('', '_prev'), how = 'left'))

    #print(new_qb_df.sample(20)) # Print a random sample of 20 rows of data

    print("Creating PNG files graphing QB stats from the previous year by touchdowns from this year...")

    overwrite_png_prev = ""
    png_prev_path = f"{project_dir}/usable_data/qb/touchdowns_and_"
    png_prev_x = 'touchdown'
    png_prev_y = ['touchdown_prev','yards_gained_prev', 'complete_pass_prev', 'pass_prev', 'interception_prev','sack_prev']
    ask_to_overwrite_graph(overwrite_png_prev, new_qb_df, png_prev_path, png_prev_x, png_prev_y)

    '''

        Creating the machine learning model

    '''

    features = ['pass_prev', 'complete_pass_prev', 'interception_prev', 'sack_prev', 'yards_gained_prev', 'touchdown_prev'] # These are the stats we will use to predict the target stat
    target = 'touchdown' # This is what we want to predict

    model_data = (new_qb_df.dropna(subset = features + [target])) # Removes null values
    train_data = (model_data.loc[(model_data['season'] == 2022) | (model_data['season'] == 2020) | (model_data['season'] == 2019) | (model_data['season'] == 2018) | (model_data['season'] == 2017) | (model_data['season'] == 2015) | (model_data['season'] == 2014) | (model_data['season'] == 2013) | (model_data['season'] == 2012) | (model_data['season'] == 2011) | (model_data['season'] == 2010) | (model_data['season'] == 2009) | (model_data['season'] == 2008) | (model_data['season'] == 2007) | (model_data['season'] == 2006) | (model_data['season'] == 2005) | (model_data['season'] == 2003) | (model_data['season'] == 2002) | (model_data['season'] == 2001) | (model_data['season'] == 2000) | (model_data['season'] == 1999)]) # Train the model on 2019-2022 data
    test_data = (model_data.loc[(model_data['season'] == 2023) | (model_data['season'] == 2021) | (model_data['season'] == 2016) | (model_data['season'] == 2012) | (model_data['season'] == 2004)]) # Test the model on 2023 data

    model = LinearRegression() # Initialize the linear regression model

    model.fit(train_data.loc[:, features], train_data[target]) # Train (fit) the model on the training data

    predictions = model.predict(test_data.loc[:, features]) # Predict using the test data
    predictions = pd.Series(predictions, index = test_data.index) # Set the index to match correct rows
    test_data['predictions'] = predictions # Join the predictions back to the dataset

    # Print some statistics to see how accurate the prediction is
    rmse = mean_squared_error(test_data['touchdown'], test_data['predictions'])**0.5
    r2 = pearsonr(test_data['touchdown'], test_data['predictions'])[0]**2
    print(f"rmse: {rmse}\nr2: {r2}")

    print("Creating PNG file graphing touchdowns by predicted touchdowns...")

    overwrite_png_predictions = ""
    png_predictions_path = f"{project_dir}/usable_data/qb/touchdowns_and_"
    png_predictions_x = 'touchdown'
    png_predictions_y = 'predictions'
    ask_to_overwrite_graph(overwrite_png_predictions, test_data, png_predictions_path, png_predictions_x, png_predictions_y)

    # Print out data of top-10 QBs by touchdowns
    #print(test_data.loc[:, ['season', 'passer_id', 'passer', 'touchdown', 'predictions']].sort_values('touchdown', ascending=False).head(20))

    print("Creating CSV file with QB stats and predicted touchdowns...")

    overwrite_csv_predictions = ""
    csv_predictions_filename = f"{project_dir}/usable_data/qb/stats_by_year_predictions.csv"
    ask_to_overwrite_csv(overwrite_csv_predictions, test_data, csv_predictions_filename)

def ask_to_overwrite_csv(user_input, dataset, path):
    while user_input.upper() != "Y" and user_input.upper() != "N":
        if os.path.isfile(path):
            print("A file with this name already exists")
            user_input = input("Would you like to overwrite the current file? Y/N: ")

            if user_input.upper() == "Y":
                dataset.to_csv(path)
            elif user_input.upper() == "N":
                print("Ok, the current file will not be overwritten")
            else:
                print("Please enter either Y or N")

        else:
            user_input = "Y"
            dataset.to_csv(path)

def generate_png_graph(x, y, dataset, path, save = True):
    sns.regplot(data = dataset, x = x, y = y)
    plt.title(f"touchdowns and {y}")

    if save == True:
        plt.savefig(f"{path}{y}.png") # Uncomment this if you want to save the figures as PNG files
    else:
        plt.show() # Uncomment this if you want a window to pop up with the graph
    
    plt.close()

def ask_to_overwrite_graph(user_input, dataset, path, x, y, save = True):
    while user_input.upper() != "Y" and user_input.upper() != "N":
        if os.path.isfile(path):
            print("A file with this name already exists")
            user_input = input("Would you like to overwrite the current file(s)? Y/N: ")

            if user_input.upper() == "Y":
                if type(y) is list:
                    for y in y:
                        generate_png_graph(x, y, dataset, path, save)
                elif type(y) is str:
                    generate_png_graph(x, y, dataset, path, save)
            elif user_input.upper() == "N":
                print("Ok, the current file will not be overwritten")
            else:
                print("Please enter either Y or N")

        else:
            user_input = "Y"
            if type(y) is list:
                for y in y:
                    generate_png_graph(x, y, dataset, path, save)
            elif type(y) is str:
                generate_png_graph(x, y, dataset, path, save)

if __name__ == "__main__":
    main()