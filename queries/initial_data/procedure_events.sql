SELECT 
    subject_id,
    hadm_id AS admission_id,
    stay_id,
    itemid,
    starttime,
    `value`
FROM procedure_events