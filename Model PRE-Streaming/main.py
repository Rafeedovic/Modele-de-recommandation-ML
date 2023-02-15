import psycopg2
from psycopg2 import Error
import sqlite3
from sqlalchemy import create_engine
from psycopg2 import extras 
from sqlalchemy.exc import SQLAlchemyError
import time
import threading
from threading import Thread
import asyncio
import pandas as pd
import numpy as np
import random
import cProfile
import pstats

from model import merging_tables
from model import create_Users_History
from model import create_matrix
from model import pre_model_item_to_item
from model import pre_model_user_to_user
from model import Model
from model import extract
from model import load
from model import analyse_etl
from model import DB_simulation

#Parametres
Number_of_Books = 5000
Number_of_users = 5000
Number_of_orders = 5000

#DB Connection + Mise place Table Books + Mise en place Table Orders
PGDATABASE="railway"
PGHOST="containers-us-west-31.railway.app"
PGPASSWORD="AqNNV4f8AaQ2bvhTXHi5"
PGPORT="7304"
PGUSER="postgres"
DATABASE_URL=f"postgresql://{ PGUSER }:{ PGPASSWORD }@{ PGHOST }:{ PGPORT }/{ PGDATABASE }"

#Current user
User_ID = random.randint(0,Number_of_users)
    

def main():
    #LET'S GOOO
    try:
        #DB_simulation(Number_of_Books,Number_of_users,Number_of_orders,DATABASE_URL)

        connection = psycopg2.connect(user=PGUSER,password=PGPASSWORD,host=PGHOST,port=PGPORT,database=PGDATABASE)
        cursor = connection.cursor()

        #EXTRACT
        table_Book, table_Order , time_extraction = extract(cursor)

        #Transform
        print("-----------------------------------------------------------Transformation-----------------------------------------------------------")
        time_transform_begin = time.perf_counter()
        df = merging_tables(table_Order,table_Book,table_Order.shape[0])
        matrix_norm = create_matrix(df)
        Users_History = create_Users_History(df)
        with cProfile.Profile() as pr:
            Top_5_books_to_recommand = Model(df,matrix_norm,Users_History,User_ID)
        time_transform_end = time.perf_counter()
        stats = pstats.Stats(pr)
        stats.sort_stats(pstats.SortKey.TIME)
        stats.print_stats(10)
        stats.dump_stats(filename='stats.prof')
        time_transform = (time_transform_end - time_transform_begin)

        #LOAD
        time_load = load(connection,Top_5_books_to_recommand,User_ID)

        #Analyse de l'ETL
        analyse_etl(time_extraction,time_transform,time_load)
        
    except (Exception, Error) as error:
        print("Error while connecting to PostgreSQL", error)

    finally:
        if connection:
            #cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")


if __name__=="__main__":
    main()
