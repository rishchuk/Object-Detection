from datetime import datetime

import pymysql
from dotenv import load_dotenv


class DB:
    def __init__(self, config):
        self.connection = pymysql.connect(**config)
        self.cursor = self.connection.cursor()
        self.create_tables()

    def create_tables(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS videos (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                upload_time DATETIME NOT NULL
            );
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS trajectories (
                id INT AUTO_INCREMENT PRIMARY KEY,
                model_name VARCHAR(25),
                object_id INT,
                object_class VARCHAR(25),
                x INT,
                y INT,
                timestamp DATETIME,
                video_id INT
            );
        ''')

    def register_video(self, video_name):
        query = '''
            INSERT INTO videos (name, upload_time)
            VALUES (%s, %s)
        '''
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.cursor.execute(query, (video_name, timestamp))
        self.connection.commit()
        return self.cursor.lastrowid

    def get_video_id(self, video_name):
        self.cursor.execute('''
            SELECT id FROM videos WHERE name = %s
        ''', (video_name,))
        result = self.cursor.fetchall()
        return result[-1] if result else None

    def add_trajectory(self, model_name, object_id, object_class, x, y, timestamp, video_id):
        self.cursor.execute('''
            INSERT INTO trajectories (model_name, object_id, object_class, x, y, timestamp, video_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        ''', (model_name, object_id, object_class, x, y, timestamp, video_id))
        self.connection.commit()

    def select_trajectories(self, video_id):
        self.cursor.execute('''
            SELECT object_id, object_class, x, y, timestamp
            FROM trajectories
            WHERE video_id = %s
            ORDER BY object_id, timestamp
        ''', (video_id,))
        return self.cursor.fetchall()

    def close(self):
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()


load_dotenv()
