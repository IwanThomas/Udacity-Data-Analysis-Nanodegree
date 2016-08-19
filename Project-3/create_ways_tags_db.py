import sqlite3
import csv
from pprint import pprint

sqlite_file = 'londonmap.db'    # name of the sqlite database file

# Connect to the database
conn = sqlite3.connect(sqlite_file)

# Get a cursor object
cur = conn.cursor()

cur.execute('''DROP TABLE IF EXISTS ways_tags;''')
conn.commit()

# Create the table, specifying the column names and data types:
cur.execute('''
    CREATE TABLE IF NOT EXISTS ways_tags(id INTEGER, key TEXT, 
    value TEXT, type TEXT, FOREIGN KEY (ID) REFERENCES nodes(id))
''')
conn.commit()

# Read in the csv file as a dictionary, format the
# data as a list of tuples:
with open('ways_tags.csv','rb') as fin:
    dr = csv.DictReader(fin) # comma is default delimiter
    to_db = [(i['id'].decode('utf-8'), i['key'].decode('utf-8'), i['value'].decode('utf-8'), i['type'].decode('utf-8')) for i in dr]
    

# insert the formatted data
cur.executemany("INSERT INTO ways_tags(id, key, value, type) VALUES (?, ?, ?, ?);", to_db)
conn.commit()

conn.close()
