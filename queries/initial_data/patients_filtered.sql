SELECT 
    a.subject_id,
    a.admission_id,
    a.admittime,
    p.gender,
    p.anchor_age,
    CASE 
        WHEN p.dod IS NOT NULL 
             AND p.dod >= a.admittime 
             AND p.dod <= date_add(a.admittime, 30)
        THEN p.dod
        ELSE NULL
    END AS dod_within_30_days
FROM admit_date_hospital a
JOIN patients p
  ON a.subject_id = p.subject_id