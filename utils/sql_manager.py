from pathlib import Path

class SQLManager:
    def __init__(self, queries_dir='queries'):
        self.queries_dir = Path(queries_dir)
    
    def load_query(self, filename):
        """Carga un archivo SQL"""
        filepath = self.queries_dir / filename
        with open(filepath, 'r') as f:
            return f.read()
    
    def execute(self, spark, filename):
        """Carga y ejecuta query"""
        query = self.load_query(filename)
        return spark.sql(query)