from shemaFuncs import *
import re
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("darkgrid")
# sns.set_context("paper")
sns.set(rc={'figure.figsize':(20, 16)})
sns.set_context("talk")
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 20)

globalColumnNames = ['indexID', 'text', 'version', 'book', 'chapter', 'verse', 'isHeb', 'isEng']

hebDfName = './data/hebDf.pkl'
engDfName = './data/engDf.pkl'

###################################################
####
# load in hebrew torah texts and metadata; create a DF
heb_text, heb_version, heb_book, heb_chapter, heb_verse = loadHebTorah()
heb_version = ['Hebrew Text'] * len(heb_version)
heb_text = np.array(heb_text)
heb_version = np.array(heb_version)
heb_book = np.array(heb_book)
heb_chapter = np.array(heb_chapter)
heb_verse = np.array(heb_verse)
hebIndex = [ b+'.'+str(c)+'.'+str(v) for b, c, v in zip(heb_book, heb_chapter, heb_verse)]
hebIndex = np.array(hebIndex)
heb_isHeb = [1] * len(hebIndex)
heb_isHeb = np.array(heb_isHeb)
heb_isEng = [0] * len(hebIndex)
heb_isEng = np.array(heb_isEng)

hebDf = pd.DataFrame(np.vstack([hebIndex, 
                      heb_text, 
                      heb_version, 
                      heb_book, 
                      heb_chapter, 
                      heb_verse, 
                      heb_isHeb, 
                      heb_isEng]).T,
                     columns = globalColumnNames)
print("\n***here is the hebrew dataframe:\n")
print(hebDf.head())

####
## load in english torah texts and metadata; create a DF

# first text 
_, eng_text = loadTest()
# remove HTML tags
removeTags = lambda x: re.sub(r'<.*?>', '', x)
vecRemoveTags = np.vectorize(removeTags)
eng_text = vecRemoveTags(eng_text)
# remove non-alpha characters
cleanEnglish = lambda x: re.sub(r'[^A-Za-z ]+', '', x)
vecCleanEnglish = np.vectorize(cleanEnglish)
eng_text = vecCleanEnglish(eng_text)
# make all lower case
allLower = lambda x: x.lower()
vecAllLower = np.vectorize(allLower)
eng_text = vecAllLower(eng_text)

# load in english torah metadata
_, eng_metaData, = loadBookVersions()

eng_version, eng_version_book, eng_version_chapter, eng_version_verse = eng_metaData
eng_version = np.array(eng_version)
eng_version_book = np.array(eng_version_book)
eng_version_chapter = np.array(eng_version_chapter)
eng_version_verse = np.array(eng_version_verse)
eng_isHeb = [0] * len(eng_text)
eng_isHeb = np.array(eng_isHeb)
eng_isEng = [1] * len(eng_text)
eng_isEng = np.array(eng_isEng)
engIndex = [ b+'.'+str(c)+'.'+str(v) for b, c, v in zip(eng_version_book, eng_version_chapter, eng_version_verse)]
engIndex = np.array(engIndex)

engDf = pd.DataFrame(np.vstack([engIndex, 
                      eng_text, 
                      eng_version, 
                      eng_version_book, 
                      eng_version_chapter, 
                      eng_version_verse, 
                      eng_isHeb, 
                      eng_isEng]).T,
                     columns = globalColumnNames)
print("\n***here is the english dataframe:\n")
print(engDf.head())

# create a pivoted engDf, to match the rows of the hebrewDf.
# save it as copy, and make another copy, to add stuff to it. 
# make a copy of the hebrew too.

engDfPivoted = engDf.pivot(index='indexID', columns='version', values='text').copy()
engDfPivotedCopy = engDfPivoted.copy()
hebDfCopy = hebDf.copy
print("\n***here is the pivotedenglish dataframe:\n")
print(engDfPivoted.head())

hebDf.to_pickle(hebDfName)
engDfPivoted.to_pickle(engDfName)
