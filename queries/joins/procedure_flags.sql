SELECT 
    subject_id,
    admission_id,
    stay_id,
    COALESCE(MAX(CASE WHEN itemid = 787541 THEN 1 ELSE 0 END), 0) as is_mechanical_ventilation,
    COALESCE(MAX(CASE WHEN itemid = 740477 THEN 1 ELSE 0 END), 0) as is_crrt,
    COALESCE(MAX(CASE WHEN itemid = 704730 THEN 1 ELSE 0 END), 0) as has_arterial_line,
    COALESCE(MAX(CASE WHEN itemid = 753461 THEN 1 ELSE 0 END), 0) as is_intubated,
    COALESCE(MAX(CASE WHEN itemid = 724004 THEN 1 ELSE 0 END), 0) as is_prone_position,
    COALESCE(MAX(CASE WHEN itemid = 755401 THEN 1 ELSE 0 END), 0) as has_chest_tube,
    COALESCE(MAX(CASE WHEN itemid = 772190 THEN 1 ELSE 0 END), 0) as is_dnr,
    COALESCE(MAX(CASE WHEN itemid = 709168 THEN 1 ELSE 0 END), 0) as has_central_line,
    COALESCE(MAX(CASE WHEN itemid = 704890 THEN 1 ELSE 0 END), 0) as has_hemodialysis,
    COALESCE(MAX(CASE WHEN itemid = 792843 THEN 1 ELSE 0 END), 0) as has_niv,
    COALESCE(MAX(CASE WHEN itemid = 736876 THEN 1 ELSE 0 END), 0) as is_ecmo
FROM procedure_events_adm
GROUP BY subject_id, admission_id, stay_id
