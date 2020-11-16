#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 12:09:36 2020

@author: jcosme
"""
import numpy as np
import pandas as pd
import pickle


hebDfName = './data/hebDf.pkl'
engDfName = './data/engDf.pkl'


hebDf = pd.read_pickle(hebDfName)
engDfPivoted = pd.read_pickle(engDfName)

def showEngVerses(aVerse):
    subDfVals = engDfPivoted[ engDfPivoted.index == aVerse ].values[0]
    subDfCols = engDfPivoted[ engDfPivoted.index == aVerse ].columns
    for i, val in enumerate(subDfVals):
        print("\n=====================================")
        print(subDfCols[i])
        print("-------------------------------------")
        print(val)

def showHebVerse(aVerse):
    subDfVals = hebDf[ hebDf.indexID == aVerse ].text.values[0]
    print("\n\n=====================================")
    print(aVerse)
    print("-------------------------------------")
    print(subDfVals)
    print('\n')

def getRandoVerse():
    randoID = np.argmax(np.random.uniform(size=len(hebDf['indexID'])))
    return hebDf.iloc[randoID]['indexID']

def loadHebDef():
    return hebDf

def loadEngDef():
    return engDfPivoted