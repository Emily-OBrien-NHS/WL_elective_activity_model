import pandas as pd
import os
import numpy as np

specialty_parameters = pd.read_csv(r'./Model Parameters by Specialty.csv')

patients = []
occupancy = []

for code, specialty in specialty_parameters[['Specialty Code', 'Specialty']].values:
    #read in the data for that specialty
    pat = pd.read_csv(f'./Outputs/{specialty} Patients.csv')
    occ = pd.read_csv(f'./Outputs/{specialty} Occupancy.csv')

    #Patient aggregation
    pat['Theatre Wait Time'] = pat['Enter Theatre Time'] - pat['Theatre Requested Time']
    pat['Bed LoS'] = pat['Discharge Time'] - pat['Enter Bed Time']
    pat['Total Time in Model'] = pat['Discharge Time'] - pat['Hospital Arrival Time']
    pat_group = (((pat.groupby(['Run', 'Patient Type'], as_index=False)
                 .agg({'Hospital Arrival Time':'count',
                       'Theatre Required':'sum',
                       'Theatre Time Required':'mean',
                       'Theatre Wait Time':'mean',
                       'Bed LoS':'mean',
                       'Total Time in Model':'mean'}))
                 .rename(columns={'Hospital Arrival Time':'Count of Patients',
                                  'Theatre Required':'Number of Theatre Patients',
                                  'Theatre Time Required':'Average Theatre Time Required',
                                  'Theatre Wait Time':'Average Wait for Theatre',
                                  'Bed LoS':'Average Bed LoS',
                                  'Total Time in Model':'Average Time in Model'}))
                 .groupby('Patient Type', as_index=False).mean()).drop('Run', axis=1)#group again for multiple runs
    #Get the counts of patients that are still waiting for theatre at the end of the model
    #merge onto results
    still_waiting = pat.loc[~(pat['Theatre Requested Time'].isna())
                            & (pat['Enter Theatre Time'].isna()),
                            'Patient Type'].value_counts().reset_index()
    still_waiting.columns = ['Patient Type', 'Patients Still Waiting for Theatre']
    pat_group = pat_group.merge(still_waiting, on='Patient Type', how='inner')
    pat_group.insert(0, 'Specialty', specialty)
    pat_group.insert(1, 'Specialty Code', code)
    patients.append(pat_group)

    #Occupancy aggregation
    occ_group = occ.groupby('Run').agg({'No. Beds Occupied':['mean', 'max'],
                                        'Pre Theatre Queue':['mean', 'max']})
    occ_group.columns = ['Average Beds Occupied', 'Max Beds Occupied',
                         'Average Theatre Queue Length',
                         'Max Theatre Queue Length']
    #Aggregate again if multiple runs
    occ_group.agg({'Average Beds Occupied':'mean',
                   'Max Beds Occupied':'max',
                   'Average Theatre Queue Length':'mean',
                   'Max Theatre Queue Length':'max'})
    occ_group.insert(0, 'Specialty', specialty)
    occ_group.insert(1, 'Specialty Code', code)
    occupancy.append(occ_group)

patients = pd.concat(patients)
occupancy = pd.concat(occupancy)
patients.to_csv('./Overall Patient Results.csv')
occupancy.to_csv('./Overall Occupancy Results.csv')
