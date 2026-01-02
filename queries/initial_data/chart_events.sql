SELECT
    subject_id,
    hadm_id AS admission_id,
    stay_id,
    itemid,
    charttime,
    `value`
FROM chart_events
