import sqlite3
import datetime
# Create database if not Exist
conn = sqlite3.connect('database/data_monitoring.db')

# Pointer to db
curs = conn.cursor()

#db_admin_user_table = """CREATE TABLE IF NOT EXISTS admin_user(userID INTEGER, userName TEXT, userPassword TEXT, userAuth CHR(4))"""

db_session_report = """CREATE TABLE IF NOT EXISTS session_report (id INTEGER PRIMARY KEY, filename TEXT, date_time TEXT)"""

db_clip_report = """CREATE TABLE IF NOT EXISTS clip_report (id INTEGER PRIMARY KEY, filename TEXT, date_time TEXT, CATEGORY CHR(10))"""

#db_table_history_validation = "CREATE TABLE IF NOT EXISTS history_validated (report_ID INTEGER, date_time TEXT, time_in TEXT," \
#                            "time_out TEXT, prediction TEXT, truth_analysis TEXT)"

#curs.execute(db_admin_user_table)
curs.execute(db_session_report)
curs.execute(db_clip_report)
#curs.execute(db_table_history_validation)

#input_group_data = [(0, "asdwiu.mp4", "2025-12-10"), (1, "asdwiu.mp4", "2025-12-10"), (2, "asdwiu.mp4", "2025-12-10"),
#                    (3, "asdwiu.mp4", "2025-12-10"), (4, "asdwiu.mp4", "2025-12-10"), (5, "asdwiu.mp4", "2025-12-10")]
#inputdata = ("clip_20250423_191154.mp4", "2025-04-02 18:02:55", "Panic")
#inputdata1 = ("clip_2025-04-02 180632.mp4", "2025-04-02 18:06:32", "Fainting")
#inputdata2 = ("clip_2025-04-02 181048.mp4", "2025-04-02 18:10:48", "Brawl")
# To delete
#args = "DELETE FROM"
#curs.executemany("INSERT INTO session_report VALUES(?, ?, ?)", input_group_data)
#curs.execute("INSERT INTO clip_report(filename, date_time, category) values {}".format(inputdata))
#curs.execute("INSERT INTO clip_report(filename, date_time, category) values {}".format(inputdata1))
#curs.execute("INSERT INTO clip_report(filename, date_time, category) values {}".format(inputdata2))

strng = "SELECT * FROM clip_report"
curs.execute(strng)
data = curs.fetchall()
#print("The dead is comimg back: ", data)

conn.commit()
conn.close()