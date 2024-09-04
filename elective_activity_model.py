import math
import simpy
import random
import datetime
import pandas as pd
import numpy as np
from functools import reduce
from stqdm import stqdm
from sqlalchemy import create_engine
from time import strftime, gmtime
import os

#TO DO: 
#-Get average theatre LoS data from Sophie
#-Do some specialties have more than 1 theatre? if daily theatre mins > theatre open hours?
#-Are theatre open hours correct?
#-Gastroenterology only had 450 mins of theatre over the year.  Should we count
  #this as no gastro theatre? currently having 1 min theatre per day which causes 
  #queues/runs slow.
#-What percentage of in patients don't go to theatre? need data from Sophie

class default_params():

    def convert_to_gmtime(time):
        #convert times to gmt to split up wday, hour and min
        return gmtime(datetime.timedelta(minutes=time * 60).total_seconds())
    
            #########################Parameters########################
    #run times, iterations and sample times
    run_time = 60*24*365*3 #3 years of run time
    #run_time = 60*24*7
    iterations = 1
    sample_time = 120
    #Other params/resources
    theatre_LoS = 120
    no_beds = np.inf
    no_pre_theatre = 1
    #split probabilities
    IP_to_theatre = 0.6
    #empty lists for results
    pat_res = []
    occ_res = []

            #############theatre opening hours (24 hours)############
    theatre_open_time = 8.5
    theatre_close_time = 17.5
    theatre_open_ts = convert_to_gmtime(theatre_open_time)
    theatre_close_ts = convert_to_gmtime(theatre_close_time)

            ########################Arrivals#########################
    #Read in arrivals data for the 3 year period
    arr_file = 'G:/PerfInfo/Performance Management/PIT Adhocs/2024-2025/'\
               'Sophie 2425/Model Data/Activity Required.xlsx'
    #read in the sheet with each years arrivals, rename cols and append to list
    arrivals = []
    for i, sheet in enumerate(['2526', '2627','2728']):
        df = pd.read_excel(arr_file, sheet_name=sheet, usecols=[0, 1, 2, 3])
        df.columns = ['Specialty Code', 'Specialty',
                      f'IP year {i+1}', f'DC year {i+1}']
        arrivals.append(df)
    #merge together into one data frame
    arrivals = reduce(lambda x, y:
                      pd.merge(x, y, on=['Specialty Code', 'Specialty']),
                      arrivals)
    #convert columns of yearly arrivals into inter-arrival times
    IP_DC_cols = [i  for i in arrivals.columns if ('IP' in i) or ('DC' in i)]
    for col in IP_DC_cols:
        arrivals[col] = ((365*24*60) / arrivals[col]).replace(np.inf, np.nan)

            #####################Theatre Times######################
    cl3_engine = create_engine('mssql+pyodbc://@cl3-data/DataWarehouse?'\
                               'trusted_connection=yes&driver=ODBC+Driver+17'\
                               '+for+SQL+Server')
    theatre_times_sql = """
    SELECT [Specialty Code] = theatlt.ListPfmgtSpecialty,
    [Theatre Minutes] = sum(listplanneddurationmins) / (52*5) -- average theatre time per day
    FROM DataReporting.theatre.vw_SessionUtilisation util
    LEFT JOIN [DataWarehouse].[Theatre].[vw_TheatreList] theatlt
    ON util.[Session Code] = theatlt.PKTheatreListId
    WHERE theatlt.ListPfmgtSpecialty in ('BS','CS','CD','CO','DM','ET','GA',
    'GY','GO','OS','MO','NS','UG','OP','OD','OR','AN','PL','UR','VS','PS','TS')
    AND util.[Session Date] BETWEEN '01-Aug-2023' AND '31-Jul-2024'-- @DTMMONTHSTART and @DTMMONTHEND
    AND ModelHospitalExclusion = 'N'
    AND TouchTimeUncappedExclusion = 'N'
    GROUP BY theatlt.ListPfmgtSpecialty
    """
    theatre_times = pd.read_sql_query(theatre_times_sql, cl3_engine)
    cl3_engine.dispose()

            #####################Length of Stay######################
    SDMart_engine = create_engine('mssql+pyodbc://@SDMartDataLive2/InfoDB?'\
						          'trusted_connection=yes&driver=ODBC+Driver+17'\
						          '+for+SQL+Server')
    LoS_sql = """
    SELECT [Specialty Code] = spect.pfmgt_spec,
    LoS = ROUND((SUM(ipact.AcuteLoS)*1.000/COUNT(*))*(24*60), 0) -- convert to minutes
    FROM infodb.dbo.vw_ipdc_fces_pfmgt fces
    LEFT JOIN InfoDB.dbo.vw_cset_specialties spect
    ON fces.local_spec = spect.local_spec
    LEFT JOIN PiMSMarts.dbo.inpatients ipact
    ON fces.prcae_refno = ipact.prcae_refno
    WHERE (fces.patcl = '1' -- inpatients only
    OR fces.patcl = '2' AND DATEDIFF(dd, fces.admit_dttm, fces.disch_dttm)>0) --failed day cases
    AND fces.spell_los > 0 --exclude 0 los
    AND fces.admet in ('11','12','13') -- Elective only
    AND fces.disch_dttm BETWEEN '01-Aug-2023'and '31-jul-2024 23:59:59'
    AND spect.nat_spec NOT IN ('199','223','290','291','331','344','345','346',
    '360','424','499','501','560','650','651','652','653','654','655','656',
    '657','658','659','660','661','662','700','710','711','712','713','715',
    '720','721','722','723','724','725','726','727','840','920') --exclude non acute specialties
    GROUP BY spect.pfmgt_spec
    """
    LoS_times = pd.read_sql_query(LoS_sql, SDMart_engine)
    LoS_times['LoS'] = LoS_times['LoS'].replace(0, np.nan)
    SDMart_engine.dispose()

            #####################Combine Outputs#####################
    #merge specialty parameters together and filter out where no arrivals
    specialty_params = (arrivals
                        .merge(LoS_times, on='Specialty Code', how='outer')
                        .merge(theatre_times, on='Specialty Code', how='outer'))
    specialty_params = specialty_params.loc[~(specialty_params[IP_DC_cols]
                                              .isna().all(axis=1))].copy()

            ###############Create Specialty Variables###############
    specialty_codes = specialty_params['Specialty Code'].to_list()
    spec = np.nan
    spec_name = np.nan
    IP = False
    DC = False
    IP_arr = np.nan
    DC_arr = np.nan
    IP_arrs = []
    DC_arrs = []
    bed_LoS = np.nan
    theatre_time = np.nan

class spawn_patient:
    def __init__(self, p_id, is_ip, ip_theatre_prob, theatre_LoS):
        self.id = p_id
        self.patient_type = ''
        #work out if the patient requires a theatre (all day case get theatre,
        # but some inpatient require bed only).
        self.theatre_required = True
        if (is_ip) and (random.uniform(0,1) <= ip_theatre_prob):
            self.theatre_required = False
        #get the amount of theatre time the patient needs, min 30 mins.
        self.theatre_time_needed = max(30,
                                       (random.expovariate(1.0 / theatre_LoS)))
        #Establish variables to store results
        self.arrival_time = np.nan
        self.theatre_priority = np.nan
        self.theatre_request_time = np.nan
        self.enter_theatre_time = np.nan
        self.leave_theatre_time = np.nan
        self.bed_request_time = np.nan
        self.enter_bed_time = np.nan
        self.discharge_time = np.nan

class elective_model:
    def __init__(self, run_number, input_params):
        #empty lists for storing results
        self.patient_results = []
        self.bed_occupancy_results = []
        #set up theatre open/closed
        self.theatre_open = False
        #start environment, set patient counter to 0 and set run number
        self.env = simpy.Environment()
        self.input_params = input_params
        self.patient_counter = 0
        self.run_number = run_number
        #establish resources
        self.bed = simpy.PriorityResource(self.env,
                                          capacity=input_params.no_beds)
        self.pre_theatre = simpy.PriorityResource(self.env,
                                                  capacity=
                                                  input_params.no_pre_theatre)
        self.theatre = simpy.Container(self.env, init=input_params.theatre_time)

    ########################ARRIVALS################################
    def generate_inpatient(self):
        yield self.env.timeout(1)
        while True:
            #Increase patient counter and spawn patient
            self.patient_counter += 1
            p = spawn_patient(self.patient_counter, True,
                              self.input_params.IP_to_theatre,
                              self.input_params.theatre_LoS)
            p.patient_type = 'In Patient'
            #Begin patient journey
            self.env.process(self.elective_activity_journey(p))
            #Randomly sample the time between patients
            sampled_interarrival = random.expovariate(1.0
                                                / self.input_params.IP_arr)
            yield self.env.timeout(sampled_interarrival)
    
    def generate_daycase(self):
        yield self.env.timeout(1)
        while True:
            #Increase patient counter and spawn patient
            self.patient_counter += 1
            p = spawn_patient(self.patient_counter, False,
                              self.input_params.IP_to_theatre,
                              self.input_params.theatre_LoS)
            p.patient_type = 'Day Case'
            #Begin patient journey
            self.env.process(self.elective_activity_journey(p))
            #Randomly sample the time between patients
            sampled_interarrival = random.expovariate(1.0
                                                / self.input_params.DC_arr)
            yield self.env.timeout(sampled_interarrival)

    ######################## PROCESS ###############################
    def elective_activity_journey(self, patient):
        #patient arrives in the hospital
        patient.arrival_time = self.env.now

        ############# THEATRE #############
        #if patient needs to go to theatre and theatre has theatre hours to give
        if ((patient.theatre_required)
            and not (math.isnan(self.input_params.theatre_time))):
            #In patients to have higher priority on theatre access
            patient.theatre_priority = (1 if patient.patient_type=='In Patient'
                                        else 2)
            #patient theatre time should not exceed a theatre day minutes
            patient.theatre_time_needed = min(patient.theatre_time_needed,
                                              self.input_params.theatre_time)
            #Request theatre space and wait for the theatre to be open with
            #enough minutes
            patient.theatre_request_time = self.env.now
            while not ((self.theatre_open)
                       and (self.theatre.level >= patient.theatre_time_needed)):
                #Record data of the patients who are queuing at the end of the model
                if math.floor(self.env.now) == (self.input_params.run_time - 1):
                    self.store_patient_results(patient)
                yield self.env.timeout(1)
            #use the pre_theatre request to have priority requests to order
            #and assign theatre time
            with (self.pre_theatre.request(priority=patient.theatre_priority)
                  as pre_theatre_req):
                yield pre_theatre_req
                #once patient has reached this point, they go straight into
                #theatre and use their theatre time, and waits in theatre
                #until this time is up
                patient.enter_theatre_time = self.env.now
                yield self.theatre.get(patient.theatre_time_needed)
                yield self.env.timeout(patient.theatre_time_needed)
                patient.leave_theatre_time = self.env.now

        ############# BEDS #############
        #Request a bed if In Patient (beds are unlimited)
        if patient.patient_type == 'In Patient':
            patient.bed_request_time = self.env.now
            with self.bed.request() as bed_req:
                yield bed_req
                patient.enter_bed_time = self.env.now
                #sample the bed LoS, take out the time the patient was in
                #theatre. 0 if no LoS data.
                sampled_bed_time = (round(random.expovariate(
                                          1.0/ self.input_params.bed_LoS)) 
                                    if not math.isnan(self.input_params.bed_LoS)
                                    else 0)
                bed_time = max(sampled_bed_time - patient.theatre_time_needed, 5)
                yield self.env.timeout(bed_time)

        ############# DISCHARGE #############
        patient.discharge_time = self.env.now
        self.store_patient_results(patient)

    ###################### CHANGE YEARLY ARRIVAL RATE ##########################
    def change_year_arr(self):
        for arrs in zip(self.input_params.IP_arrs, self.input_params.DC_arrs):
            self.input_params.IP_arr = arrs[0]
            self.input_params.DC_arr = arrs[1]
            yield self.env.timeout(self.input_params.run_time / 3)

    ###################### CLOSE AND RESTOCK THEATRE ###########################
    def close_theatre(self):
        #Process to close the theatre outside opening hours and at weekends
        while True:
            #Get current model time
            now = self.env.now
            time = gmtime(datetime.timedelta(minutes=now).total_seconds())
            #make sure it's not a weekend
            if ((time.tm_wday not in (5, 6))
                #if time is between open and close hour, then theatre is open
                and (((time.tm_hour > self.input_params.theatre_open_ts.tm_hour)
                    and (time.tm_hour < self.input_params.theatre_close_ts.tm_hour))
                #if time hour is equal to open hour, check minutes
                or ((time.tm_hour == self.input_params.theatre_open_ts.tm_hour)
                    and (time.tm_min > self.input_params.theatre_open_ts.tm_min))
                #if the time hour is equal to the close hour, check minutes
                or ((time.tm_hour == self.input_params.theatre_close_ts.tm_hour)
                    and (time.tm_min < self.input_params.theatre_close_ts.tm_min)))):
                    self.theatre_open = True
            else:
                self.theatre_open = False
            #Repeat every minute
            yield self.env.timeout(1)
    
    def add_daily_theatre_mins(self):
        #add the daily theatre minutes on each day
        while True:
            amount = self.input_params.theatre_time - self.theatre.level
            if amount > 0:
                yield self.theatre.put(amount)
            #repeat process every midnight
            yield self.env.timeout(24*60)

    ######################## RESULTS RECORDING ###############################
    def store_patient_results(self, patient):
        #Record the patient details
        self.patient_results.append([self.run_number,
                                     patient.id,
                                     patient.patient_type,
                                     patient.theatre_required,
                                     patient.theatre_time_needed,
                                     patient.arrival_time,
                                     patient.theatre_priority,
                                     patient.theatre_request_time,
                                     patient.enter_theatre_time,
                                     patient.leave_theatre_time,
                                     patient.bed_request_time,
                                     patient.enter_bed_time,
                                     patient.discharge_time])
    
    def store_occupancy(self):
        #record resource/container occupancy over time
        while True:
            self.bed_occupancy_results.append([self.run_number,
                                               self.bed._env.now,
                                               self.theatre_open,
                                               self.bed.count,
                                               self.pre_theatre.count,
                                               len(self.pre_theatre.queue),
                                               self.theatre.level])
            yield self.env.timeout(self.input_params.sample_time)

    ######################## RUN ###############################
    def run(self):
        self.env.process(self.change_year_arr())
        #Only run generators if that specialty has those patients
        if self.input_params.IP:
            self.env.process(self.generate_inpatient())
        if self.input_params.DC:
            self.env.process(self.generate_daycase())
        #Run occupancy and theatre close funtions
        self.env.process(self.store_occupancy())
        self.env.process(self.close_theatre())
        #Only restock theatre minutes if that specialty goes to theatre
        if not math.isnan(self.input_params.theatre_time):
            self.env.process(self.add_daily_theatre_mins())
        #Run for input time
        self.env.run(until=(self.input_params.run_time))
        #assign these back to default params so model can be run in parallel
        #and as is
        default_params.pat_res += self.patient_results
        default_params.occ_res += self.bed_occupancy_results
        return self.patient_results

def export_results(specialty, pat_results, occ_results):
    patient_df = (pd.DataFrame(pat_results,
                 columns=['Run', 'Patient ID', 'Patient Type',
                          'Theatre Required', 'Theatre Time Required',
                          'Hospital Arrival Time', 'Theatre Priority',
                          'Theatre Requested Time', 'Enter Theatre Time',
                          'Leave Theatre Time', 'Bed Request Time',
                          'Enter Bed Time', 'Discharge Time'])
                          .sort_values(by=['Run', 'Patient ID']))

    occ_df = (pd.DataFrame(occ_results,
              columns=['Run', 'Time', 'Theatre Open', 'No. Beds Occupied',
                       'Pre Theatre Occupancy', 'Pre Theatre Queue',
                       'Theatre Minutes Level'])
                       .sort_values(by=['Run', 'Time']))
    #write to csv
    output_path = 'C:/Users/obriene/Projects/Wait Lists/Elective Activity Model/Outputs'
    patient_df.to_csv(f'{output_path}/{specialty} Patients.csv', index=False)
    occ_df.to_csv(f'{output_path}/{specialty} Occupancy.csv', index=False)
    return patient_df, occ_df

def run_the_model(input_params):
    for spec in input_params.specialty_codes:
        #define the parameters for each specialty
        spec_params = (input_params.specialty_params.loc[
                       input_params.specialty_params['Specialty Code'] == spec]
                       .values.tolist()[0])
        input_params.spec = spec_params[0]
        input_params.spec_name = spec_params[1]
        #get list of the different arrival rates over the years and if there
        #are or aren't IP or DC
        input_params.IP_arrs =  [spec_params[i] for i in [2, 4, 6]]
        input_params.IP = not all([math.isnan(i) for i in input_params.IP_arrs])
        input_params.DC_arrs = [spec_params[i] for i in [3, 5, 7]]
        input_params.DC = not all([math.isnan(i) for i in input_params.DC_arrs])
        #Get LoS and theatre time
        input_params.bed_LoS = spec_params[8]
        input_params.theatre_time = spec_params[9]
        #run the model for that specialty for the inputted iterations
        print(f'Running model for {input_params.spec_name}')
        print(spec_params)
        #run the iterations for that specialty
        for run in range(input_params.iterations):
            print(f"Run {run+1} of {input_params.iterations}")
            model = elective_model(run, input_params)
            model.run()
        #export the results
        patient_df, occ_df = export_results(input_params.spec_name,
                                            input_params.pat_res,
                                            input_params.occ_res)
        #reset results lists for next specialty
        input_params.pat_res = []
        input_params.occ_res = []
    input_params.specialty_params.to_csv('Model Parameters by Specialty.csv',
                                         index=False)

run_the_model(default_params)
