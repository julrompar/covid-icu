SELECT 
    a.subject_id,
    a.admission_id,
    a.stay_id,
    c.itemid,
    c.value
FROM icu_stays_data a
INNER JOIN chart_events_data c
    ON a.stay_id = c.stay_id
    AND c.charttime >= a.admittime
    AND c.charttime <= date_add(a.admittime, 1)