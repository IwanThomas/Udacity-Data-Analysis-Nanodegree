import sqlite3
import csv
from pprint import pprint

sqlite_file = 'londonmap.db'    # name of the sqlite database file

# Connect to the database
conn = sqlite3.connect(sqlite_file)

# Get a cursor object
cur = conn.cursor()

cur.execute('''DROP TABLE IF EXISTS ways;''')
conn.commit()

# Create the table, specifying the column names and data types:
cur.execute('''
    CREATE TABLE IF NOT EXISTS ways(id INTEGER PRIMARY KEY,  
    user TEXT, uid INTEGER, version TEXT, changeset INTEGER, timestamp DATE)
''')
conn.commit()

# Read in the csv file as a dictionary, format the
# data as a list of tuples:
with open('ways.csv','rb') as fin:
    dr = csv.DictReader(fin) # comma is default delimiter
    to_db = [(i['id'].decode('utf-8'), i['user'].decode('utf-8'),i['uid'].decode('utf-8'),i['version'].decode('utf-8'),
              i['changeset'].decode('utf-8'),i['timestamp'].decode('utf-8')) for i in dr]
    

# insert the formatted data
cur.executemany("INSERT INTO ways(id, user, uid, version, changeset, timestamp) VALUES (?, ?, ?, ?,?, ?);", to_db)
conn.commit()


cur.execute('SELECT * FROM nodes')
all_rows = cur.fetchall()
#print('1):')
#pprint(all_rows)

conn.close()
