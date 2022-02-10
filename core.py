import sqlite3
from sqlite3 import Error
from flask import Flask, g, render_template, request

# core methods

# create a database
def get_message_db():

    # check if 'message_db' exists, if not then establish one
    if 'user_db' not in g:
        try:
            g.user_db = sqlite3.connect("./db/user_db.sqlite")
            print(sqlite3.version)
        except Error as e:
            print(e)

    cmd = \
    f"""
        CREATE TABLE IF NOT EXISTS User (
            id INT PRIMARY KEY,
        );
    """ 

    # create a table - 'Messages'
    g.message_db.cursor().executescript(cmd)
    g.message_db.close()
    
    return g.message_db
