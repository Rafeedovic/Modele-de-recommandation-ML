import pandas as pd
import time 
import numpy as np
import random
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
import asyncio
import threading
from threading import Thread
from ast import literal_eval
from psycopg2 import Error

#R_1 : pd.DataFrame()
#R_2 : pd.DataFrame()

def data_prep(df,Number_of_users):
    #Asynchrone process
    th3 = Thread(target=create_matrix,args=(df,))
    th4 = Thread(target=create_Users_History,args=(df,))
    th5 = Thread(target=creer_table_recommandation,args=(Number_of_users,))

    th3.start()
    th4.start()
    th5.start()

    th3.join()
    th4.join()
    th5.join()
    return matrix_norm,Users_History,Recommandation

def merging_tables(Order : pd.DataFrame, Book:pd.DataFrame ,Number_of_orders : int):
    df = pd.merge(Order, Book, on='Book_ID', how='inner')
    L= [1.0] * Number_of_orders
    df['bool']=L
    return df

def create_Users_History(df : pd.DataFrame):  
    global Users_History  
    Users_History = pd.DataFrame()
    Users_History['User_ID'] = df['User_ID'].unique()
    df['Book_ID']=df['Book_ID'].astype(str)
    Users_History['History'] = df[['User_ID','Book_ID']].groupby(['User_ID']).transform(lambda x: ','.join(x))
    df['Book_ID']=df['Book_ID'].astype(int)

    Users_History = Users_History.set_index('User_ID')
    #Users_History = Users_History.rename_axis('User_ID').reset_index()
    Users_History = Users_History.sort_index(ascending=True)

def create_matrix(df : pd.DataFrame):
    global matrix_norm
    def Normalisation(row):
        new_row = (row - row.mean()) / (row.max()-row.min())
        return new_row

    matrix = df.pivot_table(index='User_ID', columns='Book_Name',values='bool')
    matrix = matrix.fillna(0)
    matrix_norm = matrix.apply(Normalisation)

def creer_table_recommandation(Number_of_users : int):
    global Recommandation
    #Colonne Next_Books de la Table Order 
    a = [((-1,0),(-1,0),(-1,0),(-1,0),(-1,0))] * Number_of_users
    Recommandation = pd.DataFrame(pd.Series(a),columns=['Next_Books'])
    Recommandation = Recommandation.rename_axis('User_ID').reset_index()
    #Recommandation.columns.rename('Next_Books',inplace=True)

    Recommandation['User_ID']=Recommandation['User_ID'].astype(int)
    return Recommandation

def pre_model_item_to_item(df : pd.DataFrame, matrix_norm : pd.DataFrame,Users_History : pd.DataFrame, User_ID : int):
    if (User_ID in Users_History.index):
        #ICI Le client a déja prété au moins 1 livre
        #Cherchons le dernier livre que ce client a acheté
        L = Users_History['History'][User_ID].split(',')
        ID = int(L[len(L)-1])

        #OLD : book_users_rating = matrix_norm[matrix.columns[ID]]
        Target_Book_Name = df[df['Book_ID']==ID]['Book_Name'].to_list()[0]
        #print(Target_Book_Name)
        book_users_rating = matrix_norm[Target_Book_Name]
        similar_to_book = matrix_norm.corrwith(book_users_rating)
        similar_to_book = similar_to_book.sort_values(ascending=False)

        #OLD : similar_to_book = round(similar_to_book,6)
        similar_to_book = round(similar_to_book,5)
        a = similar_to_book.tolist()
        
        a = a[:6]

        #tuple creation
        aa=[]
        for i in range(1,len(a)):
            Book_name = similar_to_book.index[i]
            Book_IDD = df[df['Book_Name']==Book_name]['Book_ID'].to_list()[0]
            Book_Corr = a[i]
            k = (Book_IDD , Book_Corr)
            aa.append(k,)

        for i in range(6-len(a)):
            aa.append((-1,0))

        top_5_books = (*aa, )
        #print("i : ",User_ID," ",top_5_books) 
        #print("finished item-to-item")
        #print(type(top_5_books))
        return top_5_books

def item_to_item(df : pd.DataFrame, matrix_norm : pd.DataFrame, Users_History : pd.DataFrame , Recommandation : pd.DataFrame):
    print("Item-to-item is now running ...")
    global R_1
    R_1 = creer_table_recommandation(Recommandation.shape[0])
    for User_ID in range(Recommandation.shape[0]):
        if (User_ID in Users_History.index):
            top_5_books = pre_model_item_to_item(df,matrix_norm,Users_History,User_ID)
            #LOAD
            #print(type(top_5_books))
            R_1.at[User_ID,'Next_Books'] = top_5_books 
            print("i : ",User_ID," ",top_5_books)
    print("Item-to-item is finished")

def user_to_user(matrix_norm : pd.DataFrame,Users_History : pd.DataFrame, Recommandation : pd.DataFrame):
    print("User-to-user is now running ...")
    global R_2
    R_2 = creer_table_recommandation(Recommandation.shape[0])
    for User_ID in range(Recommandation.shape[0]):
        if (User_ID in Users_History.index):
            #ICI Le client a déja prété au moins 1 livre
            top_5_books = pre_model_user_to_user(matrix_norm,Users_History,User_ID)
            #print("uu",type(top_5_books))
            #LOAD
            #Recommandation['Next_Books'][User_ID] = top_5_books
            R_2.at[User_ID,'Next_Books'] = top_5_books
            print("u : ",User_ID," ",top_5_books)
    print("User-to-user is finished")

def pre_model_user_to_user(matrix_norm : pd.DataFrame,Users_History : pd.DataFrame, ID : int):
    if (ID in Users_History.index): 
        book_users_rating = matrix_norm[ID]
        similar_to_user = matrix_norm.corrwith(book_users_rating)
        similar_to_user = similar_to_user.sort_values(ascending=False)

        similar_to_user = round(similar_to_user,5)
        b = similar_to_user.tolist()
        #print(l)

        #Target USER
        History_of_target_user = Users_History['History'][similar_to_user.index[0]].split(',')
        History_of_target_user.reverse()
        #print(History_of_target_user,end='\n')
        
        most_recommanded_books = []
        
        b = b[:6]
        for i in range(1,len(b)):
            user = Users_History['History'][similar_to_user.index[i]].split(',')
            user = list(map(int, user))
            
            #ajout du composant correlation : on aura user[i] = (ID_book_user_actuel,correlation avec le user actuel)
            for j in range(len(user)):
                user[j] = (user[j],b[i])
            user.reverse() 

            #On supprime les livres en commun
            for element in user:
                actual_book = element[0]
                if actual_book in History_of_target_user:
                    user.remove(element)

            #on ajoute les livres restant 
            most_recommanded_books = most_recommanded_books + user
        
        #print(most_recommanded_books)
        bb = pd.DataFrame(most_recommanded_books, columns=['Book_ID','Correlation'])
        bb = bb.groupby('Book_ID')['Correlation'].mean().sort_values(ascending=False)
        bb = bb[:5]
        
        #tuple creation
        bbb=[]
        for i in range(0,bb.shape[0]):
            Book_IDD = bb.index[i]
            Book_Corr= bb[bb.index[i]]
            kk = (Book_IDD , Book_Corr)
            bbb.append(kk,)

        for i in range(5-len(bbb)):
            bbb.append((-1,0))

        top_5_books = (*bbb, )
        #print("u : ",ID," ",top_5_books) 
        #print("finished user-to-user")
        return top_5_books

def Model(df : pd.DataFrame, matrix_norm : pd.DataFrame,Users_History : pd.DataFrame, Recommandation : pd.DataFrame):
    global R_2
    global R_1
    if (df.shape[0]>=1):
        #OLD : if (matrix.shape[1]> (2.5 * matrix.shape[0])):
        if (matrix_norm.T.shape[0]> (2.5 * matrix_norm.T.shape[1])):
            print("Item-to-item is now running ... ")
            item_to_item(df,matrix_norm,Users_History, Recommandation)
            return R_1
            
        #OLD : elif (matrix.shape[0]> (2.5 * matrix.shape[1])):
        elif (matrix_norm.T.shape[1]> (2.5 * matrix_norm.T.shape[0])):
            print("User-to-user is now running ... ")
            user_to_user(matrix_norm.T,Users_History, Recommandation)
            return R_2

        #Asynchrone process
        th1 = Thread(target=item_to_item,args=(df,matrix_norm,Users_History, Recommandation))
        th2 = Thread(target=user_to_user,args=(matrix_norm.T,Users_History, Recommandation))

        th1.start()
        th2.start()

        th1.join()
        th2.join()
        
        print("Hybrid system is now running ... ")
        print(R_1)
        print(R_2)
        for User_ID in range(Recommandation.shape[0]):
            all = []
            #print("userrr ",User_ID)
            #print(type(R_1['Next_Books'][User_ID]))
            #print(type(R_2['Next_Books'][User_ID]))
            all = R_2['Next_Books'][User_ID] + R_1['Next_Books'][User_ID]
            #print(all)
            #print(type(R_2['Next_Books'][User_ID]))
            #l = literal_eval(R_2['Next_Books'][User_ID])
            #print(type(l))
            c = pd.DataFrame(all, columns=['Book_ID','Correlation'])

            #Re
            c = c.groupby('Book_ID')['Correlation'].mean().sort_values(ascending=False)
            c = c[:5] 

            #tuple creation
            cc=[]
            for i in range(0,c.shape[0]):
                Book_IDD = c.index[i]
                Book_Corr= c[c.index[i]]
                kkk = (Book_IDD , Book_Corr)
                cc.append(kkk,)

            for i in range(5-len(cc)):
                cc.append((-1,0))

            cc = (*cc, )
            print("m ",cc)
                
            #LOAD
            Recommandation.at[User_ID,'Next_Books'] = cc

    return Recommandation

def extract(connection):
    print("-------------------------------------------------------------Extraction-------------------------------------------------------------")
    try:
        time_extraction_begin = time.perf_counter()
        # SQL query to select TABLE Book
        cursor = connection.cursor()
        select_book__table_query = """SELECT * FROM "Book";"""
        cursor.execute(select_book__table_query)
        table_Book = pd.DataFrame(cursor.fetchall(), columns=['Book_ID','Book_Name'])

        # SQL query to select TABLE Order
        select_order__table_query = """SELECT * FROM "Order";"""
        cursor.execute(select_order__table_query)
        table_Order = pd.DataFrame(cursor.fetchall(), columns=['Order_ID','User_ID','Book_ID'])

        time_extraction_end = time.perf_counter()
        time_extraction = (time_extraction_end - time_extraction_begin)
        print("Sucessful Extract")
        return table_Book, table_Order , time_extraction
    except (Exception, Error) as error:
        print("Error while connecting to PostgreSQL", error)

def load(connection,Recommandation):
    print("--------------------------------------------------------------Loading---------------------------------------------------------------")
    time_load_begin = time.perf_counter()
    Recommandation['Next_Books'] = Recommandation['Next_Books'].astype(str)    
    R = list(Recommandation.itertuples(index=False))
    try:
        cursor = connection.cursor()
        args_str = ','.join(cursor.mogrify("(%s,%s)", tuple(x)).decode('utf-8') for x in R)
        cursor.execute("""TRUNCATE TABLE "Recommandation";INSERT INTO "Recommandation"  VALUES """ + (args_str))
        connection.commit()

    except (Exception, Error) as error:
        print("Error while connecting to PostgreSQL", error)
    """
    myengine = create_engine(DATABASE_URL)

    try:
        connection = myengine.connect()
        #Mise en place de la table Recommandation
        Recommandation = creer_table_recommandation(Recommandation.shape[0])
        Recommandation.to_sql('Recommandation', connection, if_exists='replace', index = False)
        time_load_end = time.perf_counter()
        time_load = (time_load_end - time_load_begin)
        print("Table Recommandation is sucessfully created & loaded")
        return time_load

    except SQLAlchemyError as SQLAlchemyError:
        print("Error while connecting to PostgreSQL", SQLAlchemyError)

    finally:
        if connection:
            #cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")
    """
    time_load_end = time.perf_counter()
    time_load = (time_load_end - time_load_begin)
    print("Table Recommandation is sucessfully created & loaded")
    return time_load

def analyse_etl(time_extraction,time_transform,time_load):
    print("--------------------------------------------------------------ANALYSE---------------------------------------------------------------")
    print(f"Extraction :  {time_extraction:0.4f} seconds \nTransform  :  {time_transform:0.4f} seconds \nLoad       :  {time_load:0.4f} seconds ")

    
def DB_simulation(Number_of_Books,Number_of_users,Number_of_orders,DATABASE_URL):
    def Creer_table_book(Number_of_Books : int):
        #Colonne Book_ID de la Table Book 
        Book_Book_ID = list(range(0,Number_of_Books,1))

        #Colonne Book_Name de la Table Book 
        #Book_list = ["L'Étranger","À la recherche du temps perdu","Le Procès","Le Petit Prince","La Condition humaine","Voyage au bout de la nuit","Les Raisins de la colère","Pour qui sonne le glas","Le Grand Meaulnes","L'Écume des jours","Le Deuxième Sexe","En attendant Godot","L'Être et le Néant","Le Nom de la rose","L'Archipel du Goulag","Paroles","Alcools","Le Lotus bleu","Le Journal d'Anne Frank","Tristes Tropiques","Le Meilleur des mondes","1984","Astérix le Gaulois","La Cantatrice chauve","Trois essais sur la théorie sexuelle","L'Œuvre au noir","Lolita","Ulysse","Le Désert des Tartares","Les Faux-monnayeurs","Le Hussard sur le toit","Belle du Seigneur","Cent ans de solitude","Le Bruit et la Fureur","Thérèse Desqueyroux","Zazie dans le métro","La Confusion des sentiments","Autant en emporte le vent","L'Amant de lady Chatterley","La Montagne magique","Bonjour tristesse","Le Silence de la mer","La Vie mode d'emploi","Le Chien des Baskerville","Sous le soleil de Satan","Gatsby le Magnifique","La Plaisanterie","Le Mépris","Le Meurtre de Roger Ackroyd","Nadja","Aurélien","Le Soulier de satin","Six Personnages en quête d'auteur","La Résistible Ascension d'Arturo Ui","Vendredi ou les Limbes du Pacifique","La Guerre des mondes","Si c'est un homme","Le Seigneur des anneaux","Les Vrilles de la vigne","Capitale de la douleur","Martin Eden","La Ballade de la mer salée","Le Degré zéro de l'écriture","L'Honneur perdu de Katharina Blum","Le Rivage des Syrtes","Les Mots et les Choses","Sur la route","Le Merveilleux Voyage de Nils Holgersson à travers la Suède","Une chambre à soi","Chroniques martiennes","Le Ravissement de Lol V. Stein","Le Procès-verbal","Tropismes","Journal","Lord Jim","Écrits","Le Théâtre et son double","Manhattan Transfer","Fictions","Moravagine","Le Général de l'armée morte","Le Choix de Sophie","Romancero gitano","Pietr-le-Letton","Notre-Dame des Fleurs","L'Homme sans qualités","Fureur et Mystère","L'Attrape-cœurs","Pas d'orchidées pour miss Blandish","Blake et Mortimer","Les Cahiers de Malte Laurids Brigge","La Modification","Les Origines du totalitarisme","Le Maître et Marguerite","La Crucifixion en rose","Le Grand Sommeil","Amers","Gaston Lagaffe","Au-dessous du volcan","Les Enfants de minuit"]
        #Book_Book_Name = Book_list[:Number_of_Books]

        Book_list = list(range(Number_of_Books))
        Book_Book_Name = [str(x) for x in Book_list]
        #bkk = [(Book_Book_ID[i],Book_Book_Name[i]) for i in range(len(Book_Book_ID))]
        #print(bkk)
        #Conversion en un dataFrame
        Book = pd.DataFrame(np.column_stack((Book_Book_ID,Book_Book_Name)),
                                columns=['Book_ID','Book_Name'])

        Book['Book_ID']=Book['Book_ID'].astype(int)
        Book['Book_Name']=Book['Book_Name'].astype(str)

        return Book

    def creer_table_orders(Number_of_users : int, Number_of_orders : int, Number_of_Books : int):
        #Colonne Order_ID de la Table Order 
        Order_Order_ID = list(range(0,Number_of_orders,1))

        #Colonne User_ID de la Table Order 
        temp = list(range(0,Number_of_users,1))
        Order_User_ID = random.choices(temp,k=Number_of_orders)

        #Colonne Book_ID de la Table Order 
        tempp = list(range(0,Number_of_Books,1))
        Order_Book_ID = random.choices(tempp,k=Number_of_orders)

        #Conversion en un dataFrame
        Order = pd.DataFrame(np.column_stack((Order_Order_ID,Order_User_ID,Order_Book_ID)),
                                columns=['Order_ID','User_ID','Book_ID'])
                                
        Order['Order_ID']=Order['Order_ID'].astype(int)
        Order['User_ID']=Order['User_ID'].astype(int)
        Order['Book_ID']=Order['Book_ID'].astype(int)

        return Order

    myengine = create_engine(DATABASE_URL)

    try:
        connection = myengine.connect()

        #Mise en place de la table Book
        Book = Creer_table_book(Number_of_Books)
        Book.to_sql('Book', connection, if_exists='replace', index = False)
        print("Table Book is sucessfully created & loaded")

        
        #Mise en place de la table Orders
        Order = creer_table_orders(Number_of_users,Number_of_orders,Number_of_Books)
        Order.to_sql('Order', connection, if_exists='replace', index = False)
        print("Table Order is sucessfully created & loaded")
        
        
        #Mise en place de la table Recommandation
        Recommandation = creer_table_recommandation(Number_of_users)
        Recommandation.to_sql('Recommandation', connection, if_exists='replace', index = False)
        print("Table Recommandation is sucessfully created & loaded")
        
        #cursor = connection.cursor()
        #cursor.execute("")
        #connection.commit()
        dff = pd.read_sql_table('Order', connection)

    except SQLAlchemyError as SQLAlchemyError:
        print("Error while connecting to PostgreSQL", SQLAlchemyError)

    finally:
        if connection:
            #cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")