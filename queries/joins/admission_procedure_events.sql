SELECT 
    a.subject_id,
    a.admission_id,
    a.stay_id,
    a.admittime,
    COUNT(p.itemid) as num_prod_events,    
    COLLECT_LIST(p.itemid) as prod_events_itemids,
    COLLECT_LIST(p.value) as prod_values
FROM icu_stays_data a
LEFT JOIN procedure_events_data p
    ON p.subject_id = a.subject_id
    AND p.starttime >= a.admittime
    AND p.starttime <= date_add(a.admittime, 1)
GROUP BY a.subject_id, a.admission_id, a.stay_id, a.admittime