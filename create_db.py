
import sqlite3
import pandas as pd
import os

#Move data from CSV to sqlite
def createProsperDB(year):
    #Open connection
    conn = sqlite3.connect("db/prosper.db")
    
    #Load in listing and loan data for a particular year
    listings = pd.read_csv(f"./data/{year}/Listings_{year}.csv", encoding_errors = "replace")
    loans = pd.read_csv(f"./data/{year}/Loans_{year}.csv")
    
    #Insert data into db
    listings.to_sql(f'listings_{year}', conn, index=False, if_exists='replace')
    loans.to_sql(f'loans_{year}', conn, index=False, if_exists='replace')
    
    #Close connection
    conn.close()
    return

# Import the loan-listing mapping
def loadLoanListingTable():
    # Note, this csv is processed from monthly loan level data provided by Prosper as a csv. 
    # This table is massive (51gb as of Oct. 2023 and 74,215,834 rows) so we avoid processing
    # in Python and use bash instead. The mapping was extracted using the awk command and 
    # duplicates were removed with the sort | uniq commands.
    
    conn = sqlite3.connect("db/prosper.db")
    loan_listing = pd.read_csv("./data/listing_loan_unique.csv")
    loan_listing.to_sql("loan_listing", conn, index = False, if_exists = "replace")
    
    conn.close()

if __name__ == "__main__":
    #Set workingdir
    os.chdir("/Users/andrewlee/Documents/prosper-investing/")
    
    for i in [2020,2021,2022]:
        print(i)
        createProsperDB(i) #2022
    #loadLoanListingTable()