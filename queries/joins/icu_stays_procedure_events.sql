SELECT 
    a.subject_id,
    a.admission_id,
    a.stay_id,
    c.itemid,
    c.value
FROM icu_stays_data a
INNER JOIN procedure_events c
    ON a.stay_id = c.stay_id
    AND c.starttime >= a.admittime
    AND c.starttime <= date_add(a.admittime, 1)