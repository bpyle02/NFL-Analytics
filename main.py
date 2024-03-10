import os 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

pd.options.mode.chained_assignment = None  # default='warn'

'''

    Get full path of all data files in the pbp_data folder

'''

project_dir = "." # Uses the current directory
pbp_data = os.listdir(f"{project_dir}/pbp_data") # Array of file names in pbp_data folder

#print(pbp_data) # Prints out the files in the pbp_data directory

pbp_data_full_path = ([f"""{project_dir}/pbp_data/{entry}""" for entry in pbp_data]) # Array of file names in pbp_data folder with full directory path

#print(pbp_data_full_path) # Prints out the files in the pbp_data directory with full path

'''

    Use Pandas to initialize the data frame with CSV data

'''

df = pd.DataFrame() # Create data frame variable

# Loop through each file and add it to the data frame
for file in pbp_data_full_path:
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

qb_df = (df.loc[:, qb_stats].groupby(group_by_stats, as_index = False).sum()) # Group the data and aggregate the sum

#print(qb_df.sample(20)) # Print a random sample of 20 rows of data

print("Creating CSV file with QB stats by year...")

overwrite_csv = ""

while overwrite_csv.upper() != "Y" and overwrite_csv.upper() != "N":
    if os.path.isfile(f"{project_dir}/usable_data/qb/stats_by_year.csv"):
        print("A file with this name already exists")
        overwrite_csv = input("Would you like to overwrite the current stats_by_year.csv file? Y/N: ")

        if overwrite_csv.upper() == "Y":
            qb_df.to_csv(f"{project_dir}/usable_data/qb/stats_by_year.csv")
        elif overwrite_csv.upper() == "N":
            print("Ok, the current file will not be overwritten")
        else:
            print("Please enter either Y or N")

    else:
        overwrite_csv = "Y"
        qb_df.to_csv(f"{project_dir}/usable_data/qb/stats_by_year.csv")

print("Creating PNG files graphing QB stats by touchdowns...")

overwrite_png = ""

while overwrite_png.upper() != "Y" and overwrite_png.upper() != "N":
    if os.path.isfile(f"{project_dir}/usable_data/qb/touchdowns_and_sack.png"):
        print("A file with this name already exists")
        overwrite_png = input("Would you like to overwrite the current PNG graphs? Y/N: ")

        if overwrite_png.upper() == "Y":
            # Loop through each of the specified stat categories and plot the data from the QB data frame
            for y in ['yards_gained', 'complete_pass', 'pass', 'interception','sack']:
                sns.regplot(data = qb_df, x = 'touchdown', y = y)
                plt.title(f"touchdowns and {y}")
                #plt.show() # Uncomment this if you want a window to pop up with the graph
                plt.savefig(f"{project_dir}/usable_data/qb/touchdowns_and_{y}.png") # Uncomment this if you want to save the figures as PNG files
                plt.close()
        elif overwrite_png.upper() == "N":
            print("Ok, the current files will not be overwritten")
        else:
            print("Please enter either Y or N")

    else:
        overwrite_png = "Y"
        
        # Loop through each of the specified stat categories and plot the data from the QB data frame
        for y in ['yards_gained', 'complete_pass', 'pass', 'interception','sack']:
            sns.regplot(data = qb_df, x = 'touchdown', y = y)
            plt.title(f"touchdowns and {y}")
            #plt.show() # Uncomment this if you want a window to pop up with the graph
            plt.savefig(f"{project_dir}/usable_data/qb/touchdowns_and_{y}.png") # Uncomment this if you want to save the figures as PNG files
            plt.close()

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

while overwrite_png_prev.upper() != "Y" and overwrite_png_prev.upper() != "N":
    if os.path.isfile(f"{project_dir}/usable_data/qb/touchdowns_and_sack.png"):
        print("A file with this name already exists")
        overwrite_png_prev = input("Would you like to overwrite the current PNG graphs? Y/N: ")

        # Loop through each of the specified stat categories and plot the data from the QB data frame
        if overwrite_png_prev.upper() == "Y":
            for y in ['touchdown_prev','yards_gained_prev', 'complete_pass_prev', 'pass_prev', 'interception_prev','sack_prev']:
                sns.regplot(data = new_qb_df, x = 'touchdown', y = y)
                plt.title(f"touchdowns and {y}")
                # plt.show() # Uncomment this if you want a window to pop up with the graph
                plt.savefig(f"{project_dir}/usable_data/qb/touchdowns_and_{y}.png") # Uncomment this if you want to save the figures as PNG files
                plt.close()
        elif overwrite_png_prev.upper() == "N":
            print("Ok, the current files will not be overwritten")
        else:
            print("Please enter either Y or N")

    else:
        overwrite_png_prev = "Y"
        
        # Loop through each of the specified stat categories and plot the data from the QB data frame
        for y in ['touchdown_prev','yards_gained_prev', 'complete_pass_prev', 'pass_prev', 'interception_prev','sack_prev']:
            sns.regplot(data = new_qb_df, x = 'touchdown', y = y)
            plt.title(f"touchdowns and {y}")
            # plt.show() # Uncomment this if you want a window to pop up with the graph
            plt.savefig(f"{project_dir}/usable_data/qb/touchdowns_and_{y}.png") # Uncomment this if you want to save the figures as PNG files
            plt.close()

'''

    Creating the machine learning model

'''

features = ['pass_prev', 'complete_pass_prev', 'interception_prev', 'sack_prev', 'yards_gained_prev', 'touchdown_prev'] # These are the stats we will use to predict the target stat

target = 'touchdown' # This is what we want to predict

model_data = (new_qb_df.dropna(subset = features + [target])) # Removes null values

train_data = (model_data.loc[model_data['season'] == 2022]) # Train the model on 2022 data

test_data = (model_data.loc[model_data['season'] == 2023]) # Test the model on 2023 data

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

while overwrite_png_predictions.upper() != "Y" and overwrite_png_predictions.upper() != "N":
    if os.path.isfile(f"{project_dir}/usable_data/qb/touchdowns_and_sack.png"):
        print("A file with this name already exists")
        overwrite_png_predictions = input("Would you like to overwrite the current PNG graphs? Y/N: ")

        if overwrite_png_predictions.upper() == "Y":
            # Visualize the output
            sns.regplot(data = test_data, x = 'touchdown', y = 'predictions')
            plt.title('touchdown and predictions')
            # plt.show() # Uncomment this if you want a window to pop up with the graph
            plt.savefig(f"{project_dir}/usable_data/qb/touchdowns_and_predictions.png") # Uncomment this if you want to save the figures as PNG files
            plt.close()
        elif overwrite_png_predictions.upper() == "N":
            print("Ok, the current files will not be overwritten")
        else:
            print("Please enter either Y or N")

    else:
        overwrite_png_predictions = "Y"
        
        # Visualize the output
        sns.regplot(data = test_data, x = 'touchdown', y = 'predictions')
        plt.title('touchdown and predictions')
        # plt.show() # Uncomment this if you want a window to pop up with the graph
        plt.savefig(f"{project_dir}/usable_data/qb/touchdowns_and_predictions.png") # Uncomment this if you want to save the figures as PNG files
        plt.close()




# Print out data of top-10 QBs by touchdowns
#print(test_data.loc[:, ['season', 'passer_id', 'passer', 'touchdown', 'predictions']].sort_values('touchdown', ascending=False).head(20))

print("Creating CSV file with QB stats and predicted touchdowns...")

overwrite_csv_predictions = ""

while overwrite_csv_predictions.upper() != "Y" and overwrite_csv_predictions.upper() != "N":
    if os.path.isfile(f"{project_dir}/usable_data/qb/stats_by_year_predictions.csv"):
        print("A file with this name already exists")
        overwrite_csv_predictions = input("Would you like to overwrite the current stats_by_year_predictions.csv file? Y/N: ")

        if overwrite_csv_predictions.upper() == "Y":
            test_data.to_csv(f"{project_dir}/usable_data/qb/stats_by_year_predictions.csv")
        elif overwrite_csv_predictions.upper() == "N":
            print("Ok, the current file will not be overwritten")
        else:
            print("Please enter either Y or N")

    else:
        overwrite_csv_predictions = "Y"
        test_data.to_csv(f"{project_dir}/usable_data/qb/stats_by_year_predictions.csv")