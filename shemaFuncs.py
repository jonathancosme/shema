import numpy as np
import pandas as pd
import requests
import pickle
import tensorflow_datasets as tfds
import tensorflow as tf
import time
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from itertools import compress

neviim = ["Joshua",
"Judges",
"I Samuel",
"II Samuel",
"I Kings",
"II Kings",
"Isaiah",
"Jeremiah",
"Ezekiel",
"Hosea",
"Joel",
"Amos",
"Obadiah",
"Jonah",
"Micah",
"Nahum",
"Habakkuk",
"Zephaniah",
"Haggai",
"Zechariah",
"Malachi"
]

ketuvim = ["Psalms",
"Proverbs",
"Job",
"Song of Songs",
"Ruth",
"Lamentations",
"Ecclesiastes",
"Esther",
"Daniel",
"Ezra",
"Nehemiah",
"I Chronicles",
"II Chronicles"
]

torah = ["Genesis",
"Exodus",
"Leviticus",
"Numbers",
"Deuteronomy"
]

heTitle = 'Tanach_with_Text_Only'

train_books = neviim + ketuvim
test_books = torah

apiURL = "http://www.sefaria.org/api/"

x_trainName = './data/x_train.pkl'
y_trainName = './data/y_train.pkl'
x_valName = './data/x_val.pkl'
y_valName = './data/y_val.pkl'
x_testName = './data/x_test.pkl'
y_testName = './data/y_test.pkl'

allTrainVersionsName = './data/train_versions.pkl'
allTestVersionsName = './data/test_versions.pkl'

hebTorahName = './data/heTorah.pkl'

w2vName = './data/w2v.pkl'


def getSefariaText(books):
    apiURL = "http://www.sefaria.org/api/"
    he_text = []
    en_text = []
    for i, book in enumerate(books):
        tempUrl = apiURL + "index/" + book
        r = requests.get(url = tempUrl).json()
        bookLength = r['length']
        # if i != 0:
        #     val = input("\ntype 'y' to continue to next book: ") 
        #     if val != 'y':
        #         print('stopping now')
        #         break
        print('\n\ncurrent book is: {}'.format(book))
        tempUrl = apiURL + "texts/" + book
        r = requests.get(url = tempUrl).json()
        versionNames = []
        for version_i in np.arange(0, len(r['versions'])):
            curLang = r['versions'][version_i]['language']
            print('\tcurrent language is: {}'.format(curLang))
            if curLang == 'en':
                curTitle = r['versions'][version_i]['versionTitle']
                print('\t\tcurrent version is: \n\t\t{}'.format(curTitle))
                realLan = r['versions'][version_i]['versionTitle']
                if "[" not in realLan:
                    if "Testamento" not in realLan:
                        print('\t\tthis is an english version, saving version name...')
                        versionNames.append(curTitle)
        for curVer in versionNames:
            he_temp = []
            en_temp = []
            print('\n\tgetting text for book: {}, version: {}'.format(book, curVer))
            for bookChap in np.arange(1, bookLength+1):
                print("\t\tgetting chapter {}: text...".format(bookChap))
                curTitle = curVer.replace(" ", "_")
                tempUrl = apiURL + "texts/" + book + "." + str(bookChap)  + "/en/" + curTitle
                rTemp = requests.get(url = tempUrl).json()
                heUrl = apiURL + "texts/" + book + "." + str(bookChap)  + "/he/" + heTitle
                rHe = requests.get(url = heUrl).json()
                en_vers = len(rTemp['text'])
                he_vers = len(rHe['he'])
                if en_vers != he_vers:
                    print("\t\t\tlength of verses do not match; skipping this chapter")
                else:
                    he_temp.extend(rHe['he'])
                    en_temp.extend(rTemp['text'])
            he_text.extend(he_temp)
            en_text.extend(en_temp)
    return (he_text, en_text)


def getCleanAndSaveData(valSize = 0.2, randoSeed = 1, train_books=train_books, test_books=test_books):
    hebAll, engAll = getSefariaText(train_books)
    torahHeb, torahEng = getSefariaText(test_books)
    
    np.random.seed(seed=randoSeed)
    numObs = len(hebAll)
    numObs
    randos = np.random.uniform(size=numObs)
    randos
    valmask = randos <= valSize
    valmask
    trainMask = valmask == False
    trainMask
    
    # split into train and validation set
    x_trainAll = np.array(hebAll)[trainMask]
    y_trainAll = np.array(engAll)[trainMask]
    
    x_valAll = np.array(hebAll)[valmask]
    y_valAll = np.array(engAll)[valmask]
    
    with open(x_trainName, 'wb') as handle:
        pickle.dump(x_trainAll, handle, protocol=pickle.HIGHEST_PROTOCOL)
    del handle
    
    with open(y_trainName, 'wb') as handle:
        pickle.dump(y_trainAll, handle, protocol=pickle.HIGHEST_PROTOCOL)
    del handle
    
    with open(x_valName, 'wb') as handle:
        pickle.dump(x_valAll, handle, protocol=pickle.HIGHEST_PROTOCOL)
    del handle
    
    with open(y_valName, 'wb') as handle:
        pickle.dump(y_valAll, handle, protocol=pickle.HIGHEST_PROTOCOL)
    del handle
    
    with open(x_testName, 'wb') as handle:
        pickle.dump(torahHeb, handle, protocol=pickle.HIGHEST_PROTOCOL)
    del handle
    
    with open(y_testName, 'wb') as handle:
        pickle.dump(torahEng, handle, protocol=pickle.HIGHEST_PROTOCOL)
    del handle

def loadTrainAndVal():
    with open(x_trainName, 'rb') as handle:
        x_train = pickle.load(handle)
    del handle
    with open(y_trainName, 'rb') as handle:
        y_train = pickle.load(handle)
    del handle
    with open(x_valName, 'rb') as handle:
        x_val = pickle.load(handle)
    del handle
    with open(y_valName, 'rb') as handle:
        y_val = pickle.load(handle)
    del handle
    return (x_train, y_train, x_val, y_val)
    
def loadTest(): 
    with open(x_testName, 'rb') as handle:
        x_test = pickle.load(handle)
    del handle
    with open(y_testName, 'rb') as handle:
        y_test = pickle.load(handle)
    del handle
    return (x_test, y_test)

def loadTrainAndValAsBytes():
    x_train, y_train, x_val, y_val = loadTrainAndVal()
    x_train = [ bytes(x, 'utf-8') for x in x_train.tolist() ]
    y_train = [ bytes(x, 'utf-8') for x in y_train.tolist() ]
    x_val = [ bytes(x, 'utf-8') for x in x_val.tolist() ]
    y_val = [ bytes(x, 'utf-8') for x in y_val.tolist() ]
    return (x_train, y_train, x_val, y_val)

def getSefariaVersion(books):
    apiURL = "http://www.sefaria.org/api/"
    he_text = []
    book_names = []
    chapter_number = []
    vers_number = []
    for i, book in enumerate(books):
        tempUrl = apiURL + "index/" + book
        r = requests.get(url = tempUrl).json()
        bookLength = r['length']
        # if i != 0:
        #     val = input("\ntype 'y' to continue to next book: ") 
        #     if val != 'y':
        #         print('stopping now')
        #         break
        print('\n\ncurrent book is: {}'.format(book))
        tempUrl = apiURL + "texts/" + book
        r = requests.get(url = tempUrl).json()
        versionNames = []
        for version_i in np.arange(0, len(r['versions'])):
            curLang = r['versions'][version_i]['language']
            print('\tcurrent language is: {}'.format(curLang))
            if curLang == 'en':
                curTitle = r['versions'][version_i]['versionTitle']
                print('\t\tcurrent version is: \n\t\t{}'.format(curTitle))
                realLan = r['versions'][version_i]['versionTitle']
                if "[" not in realLan:
                    if "Testamento" not in realLan:
                        print('\t\tthis is an english version, saving version name...')
                        versionNames.append(curTitle)
        for curVer in versionNames:
            he_temp = []
            en_temp = []
            book_temp = []
            chapter_temp = []
            vers_temp = []
            print('\n\tgetting text for book: {}, version: {}'.format(book, curVer))
            for bookChap in np.arange(1, bookLength+1):
                print("\t\tgetting chapter {}: text...".format(bookChap))
                curTitle = curVer.replace(" ", "_")
                tempUrl = apiURL + "texts/" + book + "." + str(bookChap)  + "/en/" + curTitle
                rTemp = requests.get(url = tempUrl).json()
                heUrl = apiURL + "texts/" + book + "." + str(bookChap)  + "/he/" + heTitle
                rHe = requests.get(url = heUrl).json()
                en_vers = len(rTemp['text'])
                he_vers = len(rHe['he'])
                if en_vers != he_vers:
                    print("\t\t\tlength of verses do not match; skipping this chapter")
                else:
                    he_temp.extend([rTemp['versionTitle']] * he_vers)
                    book_temp.extend([book] * he_vers)
                    chapter_temp.extend([bookChap] * he_vers)
                    vers_temp.extend(list(range(1, he_vers+1)))
            he_text.extend(he_temp)
            book_names.extend(book_temp)
            chapter_number.extend(chapter_temp)
            vers_number.extend(vers_temp)
    return (he_text, book_names, chapter_number, vers_number)
    
def getAllVersions(train_books=train_books, test_books=test_books):
    allTrainVersions = getSefariaVersion(train_books)
    allTestVersions = getSefariaVersion(test_books)
    with open(allTrainVersionsName, 'wb') as handle:
        pickle.dump(allTrainVersions, handle, protocol=pickle.HIGHEST_PROTOCOL)
    del handle 
    with open(allTestVersionsName, 'wb') as handle:
        pickle.dump(allTestVersions, handle, protocol=pickle.HIGHEST_PROTOCOL)
    del handle
    
def loadBookVersions():
    with open(allTrainVersionsName, 'rb') as handle:
        allTrainVersions = pickle.load(handle)
    del handle
    with open(allTestVersionsName, 'rb') as handle:
        allTestVersions = pickle.load(handle)
    del handle
    return (allTrainVersions, allTestVersions)

def getHebrewTexts(books):
    apiURL = "http://www.sefaria.org/api/"
    he_text = []
    book_names = []
    chapter_number = []
    heTitles = []
    vers_number = []
    for i, book in enumerate(books):
        tempUrl = apiURL + "index/" + book
        r = requests.get(url = tempUrl).json()
        bookLength = r['length']
        print('\n\ncurrent book is: {}'.format(book))
        heTitles_temp = []
        he_temp = []
        book_temp = []
        chapter_temp = []
        vers_temp = []
        for bookChap in np.arange(1, bookLength+1):
            print("\t\tgetting chapter {}: text...".format(bookChap))
            heUrl = apiURL + "texts/" + book + "." + str(bookChap)  + "/he/" + heTitle
            rHe = requests.get(url = heUrl).json()
            he_vers = len(rHe['he'])
            heTitles_temp.extend([rHe['versionTitle']] * he_vers)
            he_temp.extend(rHe['he'])
            book_temp.extend([book] * he_vers)
            chapter_temp.extend([bookChap] * he_vers)
            vers_temp.extend(list(range(1, he_vers+1)))
        heTitles.extend(heTitles_temp)
        he_text.extend(he_temp)
        book_names.extend(book_temp)
        chapter_number.extend(chapter_temp)
        vers_number.extend(vers_temp)
    return (he_text, heTitles, book_names, chapter_number, vers_number)

def getHebrewTorah(books=test_books):
    hebTorahText = getHebrewTexts(books)
    with open(hebTorahName, 'wb') as handle:
        pickle.dump(hebTorahText, handle, protocol=pickle.HIGHEST_PROTOCOL)
    del handle 

def loadHebTorah():
    with open(hebTorahName, 'rb') as handle:
        hebTorahText = pickle.load(handle)
    del handle
    return hebTorahText

def loadW2v():
    allw2v = pd.read_pickle(w2vName)
    return allw2v




    