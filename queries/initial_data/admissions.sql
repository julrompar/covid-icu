SELECT 
    subject_id,
    hadm_id as admission_id,
    marital_status,
    race,
    admittime,
    null as hosp_outcome 
FROM admissions