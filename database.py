import pymysql
from dotenv import load_dotenv


class DB:
    def __init__(self, config):
        self.connection = pymysql.connect(**config)
        self.cursor = self.connection.cursor()
        self.create_table()

    def create_table(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS trajectories (
                id INT AUTO_INCREMENT PRIMARY KEY,
                model_name VARCHAR(25),
                object_id INT,
                object_class VARCHAR(25),
                x INT,
                y INT,
                timestamp DATETIME
            );
        ''')

    def add_trajectory(self, model_name, object_id, object_class, x, y, timestamp):
        self.cursor.execute('''
            INSERT INTO trajectories (model_name, object_id, object_class, x, y, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s)
        ''', (model_name, object_id, object_class, x, y, timestamp))
        self.connection.commit()

    def select_trajectories(self):
        with self.connection.cursor(dictionary=True) as self.cursor:
            self.cursor.execute('''
                    SELECT object_id, object_class, x, y, timestamp
                    FROM trajectories
                    ORDER BY object_id, timestamp
                ''')
            return self.cursor.fetchall()

    def close(self):
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()


load_dotenv()
