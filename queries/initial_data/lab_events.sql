SELECT
    labevent_id,
    subject_id,
    hadm_id AS admission_id,
    itemid,
    charttime,
    valuenum as value_num
FROM lab_events