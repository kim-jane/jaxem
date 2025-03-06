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
                
                    if 'solving high-fidelity' in str(match[4]):
                        stage.append(str(match[3])+' solve')
                        
                    else:
                        stage.append(str(match[3]))
                        
                    category.append(str(match[2]))
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
    return float(df.loc[(df['stage'] == stage) & (df['category'] == category), 'time'].iloc[0])
    

Nq = [40, 80, 120, 160, 200, 240]

times = [df_times(f"output/time_chiral_Nq{n}.txt") for n in Nq]
setup_times = np.array([time_from_df(df, "setup", "single") for df in times])
train_POD_GROM_times = np.array([time_from_df(df, "train POD GROM", "single") for df in times])
train_greedy_GROM_times = np.array([time_from_df(df, "train greedy GROM", "single") for df in times])
train_POD_GROM_solve_times = np.array([time_from_df(df, "train POD GROM solve", "single") for df in times])
train_greedy_GROM_solve_times = np.array([time_from_df(df, "train greedy GROM solve", "single") for df in times])

print(times)
print(setup_times)
print(train_POD_GROM_times)
print(train_greedy_GROM_times)

ms = 4
#plt.loglog(Nq, setup_times, marker='o', linestyle='dotted', markersize=ms)
plt.loglog(Nq, train_POD_GROM_solve_times + train_POD_GROM_times, marker='o', linestyle='solid', markersize=ms, color='C0', label='Training time')
plt.loglog(Nq, train_greedy_GROM_solve_times + train_greedy_GROM_times, marker='o', linestyle='solid', markersize=ms, color='C1', label='Training time')
plt.loglog(Nq, train_POD_GROM_solve_times, marker='o', linestyle='dashed', markersize=ms, color='C0')
plt.loglog(Nq, train_greedy_GROM_solve_times, marker='o', linestyle='dashed', markersize=ms, color='C1')


plt.plot([1], [0.5], color='gray', marker='o', linestyle='dashed', markersize=ms, label='Portion spent solving high-fidelity model')

plt.xlim(37, 250)
#plt.ylim(0.8, 7)
plt.legend()
plt.show()
