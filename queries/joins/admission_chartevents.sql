SELECT 
    a.subject_id,
    a.admission_id,
    a.stay_id,
    a.admittime,
    COUNT(c.itemid) as num_chart_events,
    COLLECT_LIST(c.itemid) as chart_itemids,
    COLLECT_LIST(c.value) as chart_values
FROM icu_stays_data a
LEFT JOIN chart_events_data c
    ON a.subject_id = c.subject_id
    AND c.charttime >= a.admittime
    AND c.charttime <= date_add(a.admittime, 1)
GROUP BY a.subject_id, a.admission_id, a.stay_id, a.admittime