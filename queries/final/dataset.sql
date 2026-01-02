SELECT DISTINCT
    icu.subject_id,
    icu.admission_id,
    icu.admittime,
    icu.stay_id,
    icu.length_of_stay,
    prod_events.num_prod_events,
    prod_events.prod_events_itemids,
    prod_events.prod_values,
    chart_events.num_chart_events,
    chart_events.chart_itemids,
    chart_events.chart_values,
    lab_events.num_lab_events,
    lab_events.lab_itemids,
    lab_events.lab_values,
    patients.gender,
    patients.anchor_age,
    patients.dod_within_30_days
FROM icu_stays_data icu
LEFT JOIN procedure_events_adm prod_events
    ON icu.subject_id = prod_events.subject_id
    AND icu.admission_id = prod_events.admission_id
    AND icu.stay_id = prod_events.stay_id
LEFT JOIN chart_events_adm chart_events
    ON icu.subject_id = chart_events.subject_id
    AND icu.admission_id = chart_events.admission_id
    AND icu.stay_id = chart_events.stay_id
LEFT JOIN lab_events_adm lab_events
    ON icu.subject_id = lab_events.subject_id
    AND icu.admission_id = lab_events.admission_id
    AND icu.stay_id = lab_events.stay_id
LEFT JOIN patients_data patients
    ON icu.subject_id = patients.subject_id
