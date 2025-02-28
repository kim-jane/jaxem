import numpy as np
import matplotlib.pyplot as plt
import re
import pandas as pd


def extract_times(filename):

    with open(filename, "r") as file:

        category = []
        stage = []
        message = []
        time = []

        for line in file:
            if "- end -" in line:
                
                pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - (.*?) - (.*?) - end - (.*?) \(elapsed time: ([0-9.]+) sec\)"

                # Search the line using the pattern
                match = re.search(pattern, line)

                if match:
                    category.append(str(match[2]))
                    stage.append(str(match[3]))
                    message.append(str(match[4]))
                    time.append(float(match[5]))
                    
        return category, stage, message, time
        
        
def df_times(filename):

    category, stage, message, time = extract_times(filename)
    
    data = {'stage': stage,
            'category': category,
            'time': time}

    times = pd.DataFrame(data)
    times = times.groupby(['stage', 'category'])['time'].sum().reset_index()
    common_times = times[times['category'] == 'common'].groupby('stage')['time'].first()
    times = times.merge(common_times, on='stage', suffixes=('', '_common'), how='left')
    times.loc[times['category'] != 'common', 'time'] += times['time_common'].fillna(0)
    times = times[times['category'] != 'common'].drop(columns=['time_common'])

    return times
    
def time_from_df(df, stage, category):
    return df.loc[(df['stage'] == stage) & (df['category'] == category), 'time'].iloc[0]
    

times40 = df_times("output/time_chiral_Nq40.txt")
times56 = df_times("output/time_chiral_Nq56.txt")
times80 = df_times("output/time_chiral_Nq80.txt")
#times120 = add_times("output/time_chiral_Nq120.txt")


print("40\n", times40)
print("56\n", times56)
print("80\n", times80)
#print(times120)

time_value = time_from_df()

Nq = [40, 56, 80]
setup_times = []
train_POD_GROM_times = []
train_greedy_GROM_times = []
