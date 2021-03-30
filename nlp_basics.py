'''
SPACY
-spacy is the leading library for spacy
-it has models for different languages
'''

import spacy
nlp = spacy.load('en_core_web_sm')
doc = nlp("I love myself")

#This returns a document object that contains tokens, each token comes with additional data like lema is_stop
#lematization and stopwords are used in text preprocessing, they might also coz model to perform worse, so must use it carefully

for token in doc:
    print(str(token)+","+token.lemma_+","+str(token.is_stop))

#can do pattern matching with individual tokens, to do mtaching based on lowercased text i.e. case-insesnitive, we use attr="LOWER"
from spacy.matcher import PhraseMatcher
matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
terms = ['Apple', 'Banana','Orange']
patterns = [nlp(text) for text in terms]
matcher.add("List",patterns)
text_doc=nlp("Eric loves to eat bananas,Sarah made a delicious apple pie")
matches = matcher(text_doc)

'''
Bag of words
ML doesn't work with raw text, it needs to convert to something numeric. Simplest is one-hot encoding, each document is vector of term frequencues for each term.
For each document, count up number of times a terms occurs and place it vector.
Similar lines have similar vectors.
TF-IDF(Term frequency inverse document frequency)
It is similar to bag of words except term count is scaled by term's frequency in corpous
= termfrequency * inversedocumentfrequency
tf = count of term in d/number of words in d
df = count of term in t in document set N
idf = N/df
but it will explode if its too large, to dampen it we take log
tf-id = tf*log(N/(df+1)) +1 to avoid div by 0


SpaCy handles bag of words conversion and building a model with Textcategorizer class. It is a scpacy pipe 
for processsing and transforming tokens, There are default pipes for common operations
Can add or remove pipes to models.
Below we create a textCategorizer pipe and add it to the model
'''

nlp = spacy.blank("en")
textcat = nlp.create_pipe("text_cat", config={"exclusive_classes":True, "architecture":"bow"})
nlp.add_pipe(textcat)
textcat.add_label("ham")
textcat.add_label("spam")
'''
for each label, need to convert to form textcategoorizer requeries, dictionary of boolean values, if hamtrue and spamtrue}
WORD EMBEDDINGS
-Representing text numerically can be made better using word embedding
-It is learnt by considering the context in which word appears

WORD2VEC(Google)
-It is a embedding given by google
-it is of 2 types skipgram and cbow
-They are window based
-a nn with 2 layers
-input is window of words
-each word has 2 vectors, when it is center, when it is context

Skipgram
-predicts context words
-Window of k terms, skip one word and learn neighbouring words
-words sharing context will have closer vectprs
-better for smaller data and less frequent words

Continous bow
-predicts center word from context words
-take sentence, using surrounding words predict center
-for more data and moore frequent words

GLOVE
-gives better results
-uses  co-occurence statistics and method to max probability
'''