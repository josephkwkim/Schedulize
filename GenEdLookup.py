import numpy as np
import pandas as pd
import pickle

# Return 0 or 1 based on whether Course fulfills a General Education Requirement
def lookupGenEd(cNum, college):
    fileName = "data/Dietrich Gen Eds.csv"
    picklepath = "data\\dietrich_gen_eds.p"
    try:
        with open(picklepath,'rb') as file:
            gen_eds = pickle.load(file)
    except:
        df = pd.read_csv(fileName,names=['Dept','Num','Title','1','2'])
        gen_eds = set(df['Dept'].values)
        with open(picklepath,'wb') as file:
            pickle.dump(gen_eds,file)

    return cNum in gen_eds

'''
genEdubility = lookupGenEd(73100, "dietrich")
print("73100")
print('Is Gen Ed?:', genEdubility)
print()

genEdubility = lookupGenEd(70100, "tepper")
print("70100")
print('Is Gen Ed?:', genEdubility)
print()

genEdubility = lookupGenEd(15322, "scs")
print("15322")
print('Is Gen Ed?:', genEdubility)
print()
'''