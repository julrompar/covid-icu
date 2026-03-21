import json

def modify_notebook(path):
    with open(path, 'r') as f:
        nb = json.load(f)

    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            
            # 1. Add diagnoses_icd to the file loading cell
            if "procedure_events = spark.read.csv(" in source and "diagnoses_icd =" not in source:
                source = source.replace(
                    'procedure_events = spark.read.csv("../data/nw_icu/procedureevents.csv", header=True, inferSchema=True)',
                    'procedure_events = spark.read.csv("../data/nw_icu/procedureevents.csv", header=True, inferSchema=True)\n'
                    'diagnoses_icd = spark.read.csv("../data/nw_hosp/diagnoses_icd.csv", header=True, inferSchema=True)'
                )
            
            # 2. Add temporary view for diagnoses_icd
            if 'procedure_events.createOrReplaceTempView("procedure_events")' in source and 'diagnoses_icd.createOrReplaceTempView' not in source:
                source = source.replace(
                    'procedure_events.createOrReplaceTempView("procedure_events")',
                    'procedure_events.createOrReplaceTempView("procedure_events")\n'
                    'diagnoses_icd.createOrReplaceTempView("diagnoses_icd")'
                )

            # 3. Filter base_stays
            if "SELECT DISTINCT subject_id, admission_id, stay_id, length_of_stay" in source and "FROM icu_stays_data" in source:
                source = source.replace(
                    'SELECT DISTINCT subject_id, admission_id, stay_id, length_of_stay\n    FROM icu_stays_data',
                    "WITH RankedStays AS (\n"
                    "        SELECT i.subject_id, i.admission_id, i.stay_id, i.length_of_stay,\n"
                    "               ROW_NUMBER() OVER(PARTITION BY i.subject_id ORDER BY i.admittime ASC) as rn\n"
                    "        FROM icu_stays_data i\n"
                    "        INNER JOIN diagnoses_icd d ON i.subject_id = d.subject_id AND i.admission_id = d.hadm_id\n"
                    "        WHERE d.icd_code = 'U071'\n"
                    "    )\n"
                    "    SELECT subject_id, admission_id, stay_id, length_of_stay\n"
                    "    FROM RankedStays\n"
                    "    WHERE rn = 1"
                )

            # 4. Uncomment the writing code
            if '""" dataset_complete.createOrReplaceTempView' in source:
                source = source.replace('""" dataset_complete.createOrReplaceTempView("dataset_complete")', 'dataset_complete.createOrReplaceTempView("dataset_complete")')
                source = source.replace('dataset_complete.write.mode("overwrite").parquet("../data/processed/covid_icu_dataset.parquet") """', 'dataset_complete.write.mode("overwrite").parquet("../data/processed/covid_icu_dataset.parquet")')
            
            # update cell
            lines = source.split('\n')
            cell['source'] = [line + '\n' for line in lines[:-1]] + [lines[-1]] if len(lines) > 0 else []

    with open(path, 'w') as f:
        json.dump(nb, f, indent=1)

if __name__ == '__main__':
    modify_notebook('notebooks/dataset.ipynb')
