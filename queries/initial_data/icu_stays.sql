SELECT 
    subject_id,
    hadm_id as admission_id,
    intime as admittime,
    outtime,
    stay_id,
    los as length_of_stay
FROM icu_stays