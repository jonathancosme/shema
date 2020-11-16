# Project Shema

## Summary  

We attempt to build a Hebrew-to-English Torah translator using a Transformer model. 

Here are the python scripts for [text cleaning](textClean.py), [first set of functions](shemaFuncs.py), and [second set of functions](shemaFuncs2.py).   
Here is the python script for the [word to vector analysis](word2vec.py).  
Here is the Jupyter Notebook for the [first text analysis](textAnalysis1.ipynb).   
Here is the Jupyter Notebook for the [second text analysis](textAnalysis2.ipynb).  
Here is the python script used to [train the model](trainShemaModel.py).   
Here is a [powerpoint format](jcosme-project4.pptx) of the presentation, and here is a [pdf format](jcosme-project4.pdf) of the presentation.  


### Outline  
+ The Data
+ Word to Vector Cluster Analysis
+ English vs Hebrew Analysis
+ Model Building and Selection
+ Results
+ Further Development

## The Data

Texts were obtained from [Sefaria.org](https://sefaria.org).

We use all available English Translations (multiple versions), as well as well as the original Hebrew text.

We used the following books to train our models:
+ Joshua
+ Judges
+ I Samuel
+ II Samuel
+ I Kings
+ II Kings
+ Isaiah
+ Jeremiah
+ Ezekiel
+ Hosea
+ Joel
+ Amos
+ Obadiah
+ Jonah
+ Micah
+ Nahum
+ Habakkuk
+ Zephaniah
+ Haggai
+ Zechariah
+ Malachi
+ Psalms
+ Proverbs
+ Job
+ Song of Songs
+ Ruth
+ Lamentations
+ Ecclesiastes
+ Esther
+ Daniel
+ Ezra
+ Nehemiah
+ I Chronicles
+ II Chronicles

We reserve these books, in order to test our mode:
+ Genesis
+ Exodus
+ Leviticus
+ Numbers
+ Deuteronomy

## Word to Vector Cluster Analysis

### W2V and PCA

We first explore the All the English versions of the texts, to see if we can find any distinguisihing features. 

The Word-to-Vector was performed using skip-grams (window_size = 2), and negative sampling (num_ns = 6), with embedding dimensions of 128. 

The embeddings of the w2v output were then run through a PCA analysis, in order to reduct the dimensions down to 2. 

This is a diagram of the pipline:

![w2v pipline](images/pipline.png)

After reducting the dimensionality, we plot the results:

![w2v clusters](images/hdbscanClusters2.png)

### HDBSCAN Clustering

It apprears we have around 5 distinct clusters.     
On these results, we perform an HDBSCAN clustering with a minimum cluster size of 150.   
After performing the clustering, these our our results:

![HDBSCAN results](images/hdbscanMembers.png)

We can see that the HDBSCAN picked up the clusters very well.   
Lets explore the composition of Versions for each Cluster: 

![cluster version composition](images/clusterMemberships.png)

The bar plots show us that the HDBSCAN clustering agrees with our visual clustering. 

### Cluster Exploration

Now lets explore the individual clusters, to see if we can find any patterns.

Let's start here: 

![first cluster explore](images/zoom1.png)

In this cluster, we find words like the following:
+ yered
+ yishmael
+ sotah
+ elkanah
+ yetheth
+ le'alem
+ akkad
+ chazeiros
+ chelek
+ mishpat
+ asriel
+ kasluchim
+ migdal
+ chayyi
+ p'sal

It appears that there are many transliterated Hebrew words. That is, many Hebrew words, written out with English letters. 

Let's take a look at another cluster:

![second cluster explore](images/zoom2.png)

This cluster appears to have straight Hebrew words in the English text!

Let's take a look at one more cluster:

![third cluster explore](images/zoom3.png)

This cluster, we find words like this:
+ a
+ she
+ son
+ set
+ what 
+ was
+ if
+ fly
+ an
+ has
+ like
+ by
+ on
+ fly
+ his 
+ am 

It appears that this cluster is composed of VERY simple, plain english.

In general, I have found the following patterns:

![summary cluster explore](images/clustSummary.png)

It's almost as if the x-axis is a measure of English, and the y-axis is a measure of Hebrew.

## English vs Hebrew Analysis

Now we move on to an analysis of Hebrew text and compare it to the English text.

If one were to say the same sentence in Hebrew, and in English, it would take less words to say it in Hebrew.   
This is generally true of English when compared to most other languages. 

In order to illustrate (and confirm) this, we will calculate a variable which we will  called "complexity."   

Complexity will be defined as:  
(number of unique words in verse) / (total number of words in verse)   
As shown in this diagram:

![complexity definition](images/complexityCalc.png)

If we apply this to all the verses in all English versions, and the verses in the Hebrew version, we get the following results:

![complexity results](images/complexResults.png)

In addition, for every English Verse, there will be a set of words that all of the English version share. 

We are interesting in seeing the proportion of these "shared" word sets, to the set of unique words, for a version verse. 

here is an example calculation:

![proportion calc](images/sharedWordsCalc.png)


We perform this calculation on each verse, and plot the results: 

![proportion results](images/sharedEngl.png)

It looks like the proportion of shared words is between 30% and 50%.  

This gives us a good basis for our target accuracy: since between 30% and 50% are shared words, our translation model should at least be able to accurately predict between 30% and 50% of a verse.

## Model Building and Selection

### Iteration 1

Our first model gave very poor results:

![bad model results](images/badTrans.png)

So we switched our Hebrew text from vowelized to non-vowelized (this is how Hebrew is actually written):

![switched hebrew](images/switchedHeb.png)

After this switch, we got better results.

### Iteration 2

Here is the first model built afterwards: 

![m1-1](images/m1-1.png)

In order to try to increase the results on the validation model, we changed the max length of a sentence from 500, to 50.  
Here are the results:

![m1-2](images/m1-2.png)

No improvement. 

We decided to see if increasing the number of neurons in the feed-forward network would improve the model. Here are the results:

![m1-2](images/m1-3.png)

Performance actually got worse!

In addition, our attention head graphs STILL did not look as they should. 

### Iteration 3

We decided to change from sub-word-tokenization, to regular word-tokenization.

We saw an immediate improvement, as shown in the diagram below: 

![m2-1](images/m2-1.png)

We then tried making a very large model, to see if that would improve our validation results:

![m2-2](images/m2-2.png)

Results got worse.  
It seems the more complex the model is, the worse the validation results are. 

Since this appeared to be the case, we went ahead and made our model smaller, by reducing the number of layers from 4 to 3.  
Here are the results:

![m2-3](images/m2-3.png)

These are the best results so far! 

Next, we slightly increased model complexity, and increased the drop out rate, to see what would happen:

![m2-4](images/m2-4.png)

The results are better, but not by a significant amount. This larger model takes much more time to train, so I wouldn't use this one. 

Finally, we reduced the number of layers from 3 to 2, to see if that would give us a better fit:

![m2-5](images/m2-6.png)

The fit got worse. 

The model we decided to go with is:

![selected model](images/m2-3.png)

## Results

### Here is example 1 of 4:

![ex1-a](images/example1-a.png)
![ex1-b](images/example1-b.png)
![ex1-c](images/example1-c.png)

### Here is example 2 of 4:

![ex2-a](images/example2-a.png)
![ex2-b](images/example2-b.png)
![ex2-c](images/example2-c.png)

### Here is example 3 of 4:

![ex3-a](images/example3-a.png)
![ex3-b](images/example3-b.png)
![ex3-c](images/example3-c.png)

### Here is example 4 of 4:

![ex4-a](images/example4-a.png)
![ex4-b](images/example4-b.png)
![ex4-c](images/example4-c.png)

## Further Development

I would tune batch-size and number of epochs hyper-parameters.

In addition, I would try to build a web application that could either take in raw Hebrew texts and translate it, or have a list of Torah verses to select from and translate. 

It would be nice to see the different English Version translations, and also where the translation fits in compared to the cluster analysis and word complexity. Seeing the results of the attention heads could also be insightful

![further dev](images/furtherDev.png)
