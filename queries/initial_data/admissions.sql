SELECT 
    subject_id,
    hadm_id as admission_id,
    marital_status,
    race,
    admittime,
    dischtime,
    deathtime,
    null as hosp_outcome 
FROM admissions