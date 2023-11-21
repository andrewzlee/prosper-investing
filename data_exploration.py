
"""
@author: andrewlee

# 

"""

import sqlite3
import pandas as pd
import os
import numpy as np


#Set workingdir
os.chdir("/Users/andrewlee/Documents/prosper-investing/")

conn = sqlite3.connect("./db/prosper.db")



##############################
# Begin basic analysis
##############################
cursor = conn.cursor()

#Count rows in table
for year in years:
    query = f"""
        SELECT listing_term, count(*)
        FROM listings_{year}
        group by listing_term 
        """
    cursor.execute(query)
    result = cursor.fetchall()
    print(year)
    print(result)

#Get default rates for 3 year loans
query = """
    SELECT prosper_rating, sum(case when loan_status in (2,3) then 1 else 0 end)/(count(*)* 1.0)
    FROM loans_2022
    where term = 36
    group by prosper_rating
    """
cursor.execute(query)
result = cursor.fetchall()

# ('AA',0.03297609233305853)
# ('A', 0.07646920781789533)
# ('B', 0.10688262166345755)
# ('C', 0.15545451012397496)
# ('D', 0.20536995285919246)
# ('E', 0.2048219420482194)
# ('HR',0.2122854561878952)


#https://developers.prosper.com/docs/investor/listings-api/
# After March 31st, 2017, all new listings contain only TransUnion credit bureau data

#General info about individual who have defaulted

query = """
    SELECT list.TUFicoRange,
    --loan.prosper_rating, 
    --round(avg(months_employed),2), 
    --round(AVG(stated_monthly_income)*12,2), 
    --round(AVG(amount_funded),2),
    round(sum(case when loan_status in (2,3) then 1 else 0 end)/(count(*)* 1.0),2),
    sum(case when loan_status in (2,3) then 1 else 0 end),
    count(*)
    FROM loans_2019 loan 
    JOIN loan_listing ll
        ON loan.loan_number = ll.loanid
    JOIN listings_2019 list
        ON ll.listingnumber = list.listing_number
    WHERE loan.term = 36
    GROUP BY list.TUFicoRange --loan.prosper_rating
"""
cursor.execute(query)
result = cursor.fetchall()
print(result)

query = """
    SELECT occupation,list.prosper_rating, round(sum(case when loan_status in (2,3) then 1 else 0 end)/(count(*)* 1.0),2),
    round(avg(lender_yield),2)
    FROM loans_2019 loan 
    JOIN loan_listing ll
        ON loan.loan_number = ll.loanid
    JOIN listings_2019 list
        ON ll.listingnumber = list.listing_number
    WHERE loan.term = 36
    GROUP BY list.occupation, list.prosper_rating
"""
cursor.execute(query)
result = cursor.fetchall()

# Homeowner
query = """
    SELECT list.CoBorrowerApplication, 
    round(sum(case when loan_status in (2,3) then 1 else 0 end)/(count(*)* 1.0),2),
    count(*)
    FROM loans_2019 loan 
    JOIN loan_listing ll
        ON loan.loan_number = ll.loanid
    JOIN listings_2019 list
        ON ll.listingnumber = list.listing_number
    WHERE loan.term = 36
    GROUP BY list.CoBorrowerApplication
"""
cursor.execute(query)
result = cursor.fetchall()
print(result)


###
query = """
    SELECT distinct TUFicoRange
    FROM listings_2019 
    order by TUFicoRange
"""
cursor.execute(query)
result = cursor.fetchall()
print(result)


## Show how much each class (1) repaid, (2) borrowed, (3) percent repaid, (4) principal repaid
## Conclusion, irrespective of class, you can expect a 50% loss on your investment
## when a default occurs, on average.
query = """
    SELECT prosper_rating, round(avg(principal_paid + interest_paid),2), 
        round(avg(amount_borrowed),2), 
        round(avg((principal_paid + interest_paid)/amount_borrowed),2),
        round(avg(principal_paid/amount_borrowed),2)
    FROM loans_2019  
    WHERE term = 36 and loan_status in (2,3)
    group by prosper_rating
"""
cursor.execute(query)
result = cursor.fetchall()
print(result)

