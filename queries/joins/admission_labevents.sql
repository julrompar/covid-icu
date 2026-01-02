SELECT 
    a.subject_id,
    a.admission_id,
    a.stay_id,
    a.admittime,
    COUNT(l.labevent_id) as num_lab_events,
    COLLECT_LIST(l.itemid) as lab_itemids,
    COLLECT_LIST(l.value_num) as lab_values
FROM icu_stays_data a
LEFT JOIN lab_events_data l
    ON a.subject_id = l.subject_id
    AND l.charttime >= a.admittime
    AND l.charttime <= date_add(a.admittime, 1)
GROUP BY a.subject_id, a.admission_id, a.stay_id, a.admittime