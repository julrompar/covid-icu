SELECT 
    a.subject_id,
    a.admission_id,
    a.stay_id,
    l.itemid,
    l.value_num
FROM icu_stays_data a
INNER JOIN lab_events_data l
    ON a.admission_id = l.admission_id
    AND l.charttime >= a.admittime
    AND l.charttime <= date_add(a.admittime, 1)