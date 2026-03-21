#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import sys
os.environ['_JAVA_OPTIONS'] = '-Djava.security.manager=allow -Duser.name=julianromero'
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from pyspark.sql import SparkSession
import pyspark.pandas as ps
spark = SparkSession.builder.appName("SparkSession").getOrCreate()

from utils.sql_manager import SQLManager
sql_manager = SQLManager(queries_dir='../queries')


# In[3]:


#Archivos
hosp_adm = spark.read.csv("../data/nw_hosp/admissions.csv", header=True, inferSchema=True)
icu_stays = spark.read.csv("../data/nw_icu/icustays.csv", header=True, inferSchema=True)
patients = spark.read.csv("../data/nw_hosp/patients.csv", header=True, inferSchema=True)
chart_events = spark.read.csv("../data/nw_icu/chartevents.csv", header=True, inferSchema=True)
lab_events = spark.read.csv("../data/nw_hosp/labevents.csv", header=True, inferSchema=True)
procedure_events = spark.read.csv("../data/nw_icu/procedureevents.csv", header=True, inferSchema=True)
diagnoses_icd = spark.read.csv("../data/nw_hosp/diagnoses_icd.csv", header=True, inferSchema=True)


# In[4]:


#Vistas temporales
hosp_adm.createOrReplaceTempView("admissions")
icu_stays.createOrReplaceTempView("icu_stays")
patients.createOrReplaceTempView("patients")
chart_events.createOrReplaceTempView("chart_events")
lab_events.createOrReplaceTempView("lab_events")
procedure_events.createOrReplaceTempView("procedure_events")
diagnoses_icd.createOrReplaceTempView("diagnoses_icd")


# In[7]:


hosp_adm_data = sql_manager.execute(spark, 'initial_data/admissions.sql')
hosp_adm_data.createOrReplaceTempView("hosp_adm_data")

icu_stays_data = sql_manager.execute(spark, 'initial_data/icu_stays.sql')
icu_stays_data.createOrReplaceTempView("icu_stays_data")

chart_events_data = sql_manager.execute(spark, 'initial_data/chart_events.sql')
chart_events_data.createOrReplaceTempView("chart_events_data")

patients_data = sql_manager.execute(spark, 'initial_data/patients.sql')
patients_data.createOrReplaceTempView("patients_data")

lab_events_data = sql_manager.execute(spark, 'initial_data/lab_events.sql')
lab_events_data.createOrReplaceTempView("lab_events_data")

procedure_events_data = sql_manager.execute(spark, 'initial_data/procedure_events.sql')
procedure_events_data.createOrReplaceTempView("procedure_events_data")


# In[99]:


icu_stays.show(5)
stays_count = spark.sql("select subject_id, count(*) as num_stays from icu_stays group by subject_id order by num_stays desc")
stays_count.show(5)
icu_stays_data


# In[100]:


admit_date_hospital = sql_manager.execute(spark, 'dates/admision_time.sql')
admit_date_hospital.createOrReplaceTempView("admit_date_hospital")
admit_date_hospital.show(5)
admit_time_ps = admit_date_hospital.toPandas()
admit_time_ps


# In[101]:


patients_data = sql_manager.execute(spark, 'initial_data/patients_filtered.sql')
patients_data.createOrReplaceTempView("patients_data")
patients_data.show(5)
patients_data_ps = patients_data.toPandas()
patients_data_ps


# In[102]:


chart_events_post_admission = sql_manager.execute(spark, 'joins/admission_chartevents.sql')
chart_events_post_admission.createOrReplaceTempView("chart_events_adm")
chart_events_post_admission.show(5)


# In[103]:


lab_events_adm = sql_manager.execute(spark, 'joins/admission_labevents.sql')
lab_events_adm.createOrReplaceTempView("lab_events_adm")
lab_events_adm.show(5)
lab_events_adm_df = lab_events_adm.toPandas()
lab_events_adm_df


# In[104]:


chart_events = sql_manager.execute(spark, 'joins/icu_stays_chartevents.sql')
chart_events.createOrReplaceTempView("chart_events_icu")
lab_events = sql_manager.execute(spark, 'joins/icu_stays_labevents.sql')
lab_events.createOrReplaceTempView("lab_events_icu")
procedure_events = sql_manager.execute(spark, 'joins/icu_stays_procedure_events.sql')
procedure_events.createOrReplaceTempView("procedure_events_icu")


# In[105]:


chart_events


# In[106]:


lab_events


# In[107]:


procedure_events


# In[108]:


#86 Fº son 30 Cº y 113 Fº son 45 Cº
temperature = spark.sql("""
    select subject_id, admission_id, stay_id, max(try_cast(value as double)) as max_temp
    from chart_events_icu 
    where itemid = 323761 
        and value is not null
        and try_cast(value as double) is not null
    group by subject_id, admission_id, stay_id
""")

temperature.createOrReplaceTempView("temperature")



temperature


# In[109]:


systolic_arterial_bp_line = spark.sql("""
        select subject_id, admission_id, stay_id, min(value) as min_bp_systolic_line
        from chart_events_icu where itemid = 320050
            and value is not null
            and try_cast(value as double) is not null
            and try_cast(value as double) > 0
        group by subject_id, admission_id, stay_id""")
dystolic_arterial_bp_line = spark.sql("""
        select subject_id, admission_id, stay_id, min(value) as min_bp_diastolic_line
        from chart_events_icu where itemid = 320051
        group by subject_id, admission_id, stay_id""")

systolic_arterial_bp_line.createOrReplaceTempView("systolic_bp_line")
dystolic_arterial_bp_line.createOrReplaceTempView("dystolic_bp_line")

""" bp_line = systolic_arterial_bp_line.join(dystolic_arterial_bp_line,
    on=['subject_id', 'admission_id', 'stay_id'], how='full')
bp_line.createOrReplaceTempView("bp_line")
missing_bp_line = spark.sql("select * from bp_line where min_bp_systolic is null or min_bp_diastolic is null")
missing_bp_line """

bp_line = systolic_arterial_bp_line.join(dystolic_arterial_bp_line, 
    on=['subject_id', 'admission_id', 'stay_id'], how='full')
bp_line.createOrReplaceTempView("bp_line")

bp_line





# In[110]:


systolic_arterial_bp = spark.sql("""
        select subject_id, admission_id, stay_id, min(try_cast(value as double)) as min_bp_systolic
        from chart_events_icu where itemid = 320179
            and value is not null
            and try_cast(value as double) is not null
        group by subject_id, admission_id, stay_id""")
dystolic_arterial_bp= spark.sql("""
        select subject_id, admission_id, stay_id, min(try_cast(value as double)) as min_bp_diastolic
        from chart_events_icu where itemid = 320180
            and value is not null
            and try_cast(value as double) is not null
        group by subject_id, admission_id, stay_id""")

systolic_arterial_bp.createOrReplaceTempView("systolic_arterial_bp")
dystolic_arterial_bp.createOrReplaceTempView("dystolic_arterial_bp")


bp = systolic_arterial_bp.join(dystolic_arterial_bp,
    on=['subject_id', 'admission_id', 'stay_id'], how='full')
bp.createOrReplaceTempView("bp")


all_bp = bp_line.join(bp,
    on=['subject_id', 'admission_id', 'stay_id'], how='full')
all_bp.createOrReplaceTempView("all_bp")
all_bp


# In[111]:


dystolic_arterial_bp


# In[112]:


systolic_arterial_bp


# In[113]:


#Rango de valores del 0 al 100. Valores de menos del 92% se relacionan con hipoxemia grave. Limitar en rango 70-100? Se 'perderían' 354 mediciones con ese filtro
sp_o2 = spark.sql("""
        select subject_id, admission_id, stay_id, min(try_cast(value as double)) as min_oxygen_saturation
                from chart_events_icu where itemid = 320277
                    and value is not null
                    and try_cast(value as double) is not null
                group by subject_id, admission_id, stay_id""")

sp_o2.createOrReplaceTempView("sp_o2")

sp_o2

dataset_with_spo2 = all_bp.join(sp_o2,
    on=['subject_id', 'admission_id', 'stay_id'], how='full')
dataset_with_spo2.createOrReplaceTempView("dataset_with_spo2")
dataset_with_spo2


# In[114]:


#Rango entre 40 y 220 pulsaciones por minuto. Se 'pierden' 12 mediciones
pulse = spark.sql("""
        select subject_id, admission_id, stay_id, max(try_cast(value as double)) as max_pulse
                from chart_events_icu where itemid = 320045
                and value is not null
                and try_cast(value as double) is not null
                group by subject_id, admission_id, stay_id""")

pulse.createOrReplaceTempView("pulse")

dataset_with_pulse = dataset_with_spo2.join(pulse,
    on=['subject_id', 'admission_id', 'stay_id'], how='full')
dataset_with_pulse.createOrReplaceTempView("dataset_with_pulse")

pulse
dataset_with_pulse


# In[115]:


glucose = spark.sql("""
                    select subject_id, admission_id, stay_id, max(value_num) as max_glucose
                from lab_events_icu where itemid = 100001
                and value_num is not null
                group by subject_id, admission_id, stay_id""")

glucose.createOrReplaceTempView("glucose")

dataset_with_temperature = dataset_with_pulse.join(temperature, on=['subject_id', 'admission_id', 'stay_id'], how='full')

dataset_with_glucose = dataset_with_temperature.join(glucose,
    on=['subject_id', 'admission_id', 'stay_id'], how='full')   
dataset_with_glucose.createOrReplaceTempView("dataset_with_glucose")

glucose
dataset_with_glucose


# In[116]:


lactate = spark.sql("""
                    select subject_id, admission_id, stay_id, max(value_num) as max_lactate
                from lab_events_icu where itemid = 100031
                and value_num is not null
                group by subject_id, admission_id, stay_id""")

lactate.createOrReplaceTempView("lactate")




# In[117]:


procedure_flags = sql_manager.execute(spark, 'joins/procedure_flags.sql')
procedure_flags.createOrReplaceTempView("procedure_flags")


# In[ ]:


base_stays = spark.sql("""
    WITH RankedStays AS (
        SELECT i.subject_id, i.admission_id, i.stay_id, i.length_of_stay,
               ROW_NUMBER() OVER(PARTITION BY i.subject_id ORDER BY i.admittime ASC) as rn
        FROM icu_stays_data i
        INNER JOIN diagnoses_icd d ON i.subject_id = d.subject_id AND i.admission_id = d.hadm_id
        WHERE d.icd_code = 'U071'
    )
    SELECT subject_id, admission_id, stay_id, length_of_stay
    FROM RankedStays
    WHERE rn = 1
""")

dataset_complete = (base_stays
    .join(all_bp, on=['subject_id', 'admission_id', 'stay_id'], how='left')
    .join(sp_o2, on=['subject_id', 'admission_id', 'stay_id'], how='left')
    .join(pulse, on=['subject_id', 'admission_id', 'stay_id'], how='left')
    .join(temperature, on=['subject_id', 'admission_id', 'stay_id'], how='left')
    .join(glucose, on=['subject_id', 'admission_id', 'stay_id'], how='left')
    .join(lactate, on=['subject_id', 'admission_id', 'stay_id'], how='left')
    .join(procedure_flags, on=['subject_id', 'admission_id', 'stay_id'], how='left')
    .join(patients_data.select('subject_id', 'admission_id', 'gender', 'anchor_age', 'dod_within_30_days'), 
          on=['subject_id', 'admission_id'], how='left')
    .join(hosp_adm_data.select('subject_id', 'admission_id', 'marital_status', 'race'), 
          on=['subject_id', 'admission_id'], how='left')
)

procedure_columns = ['is_mechanical_ventilation', 'is_crrt', 'has_arterial_line', 
                     'is_intubated', 'is_prone_position', 'has_chest_tube', 
                     'is_dnr', 'has_central_line', 'has_hemodialysis', 
                     'has_niv', 'is_ecmo']

for col_name in procedure_columns:
    dataset_complete = dataset_complete.fillna({col_name: 0})

dataset_complete.createOrReplaceTempView("dataset_complete")

print(f"Total de stays en base: {base_stays.count()}")
print(f"Total de stays en dataset final: {dataset_complete.count()}")
dataset_complete.show(5)

dataset_complete.write.mode("overwrite").csv("../data/processed/covid_icu_dataset.csv", header=True)
dataset_complete.write.mode("overwrite").parquet("../data/processed/covid_icu_dataset.parquet")

dataset_complete


# In[9]:


admissions_with_valid_dod = spark.sql("""
    select * from hosp_adm_data 
    where deathtime is not null 
    and deathtime <= dischtime
    and deathtime > admittime""")

admissions_with_valid_dod


# In[ ]:




