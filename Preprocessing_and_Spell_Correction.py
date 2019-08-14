import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def convert_lower_case(data):
    return np.char.lower(data)
def remove_stop_words(data):
    stop_words = stopwords.words('english')
    words = word_tokenize(str(data))
    new_text = ""
    for w in words:
        if w not in stop_words and len(w) > 1:
            new_text = new_text + " " + w
    return new_text
def remove_punctuation(data):
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
        data = np.char.replace(data, "  ", " ")
    data = np.char.replace(data, ',', '')
    return data
def remove_apostrophe(data):
    return np.char.replace(data, "'", "")
def stemming(data):
    stemmer= PorterStemmer()   
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + stemmer.stem(w)
    return new_text
def convert_numbers(data):
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        try:
            w = num2words(int(w))
        except:
            a = 0
        new_text = new_text + " " + w
    new_text = np.char.replace(new_text, "-", " ")
    return new_text
def preprocess(data):
    data = convert_lower_case(data)
    data = remove_punctuation(data) #remove comma seperately
    data = remove_apostrophe(data)
    data = remove_stop_words(data)
    data = convert_numbers(data)
    data = stemming(data)
    data = remove_punctuation(data)
    data = convert_numbers(data)
    data = stemming(data) #needed again as we need to stem the words
    data = remove_punctuation(data) #needed again as num2word is giving few hypens and commas fourty-one
    data = remove_stop_words(data) #needed again as num2word is giving stop words 101 - one hundred and one
    return data

def spelling_correction(data,column):
    from symspellpy.symspellpy import SymSpell , Verbosity
    # maximum edit distance per dictionary precalculation
    max_edit_distance_dictionary = 2
    prefix_length = 7
    # create object
    sym_spell = SymSpell(max_edit_distance_dictionary, prefix_length)
    # load dictionary
    dictionary_path = "frequency_dictionary_en_82_765.txt"
    term_index = 0  # column of the term in the dictionary text file
    count_index = 1  # column of the term frequency in the dictionary text file
    if not sym_spell.load_dictionary(dictionary_path, term_index, count_index):
       print("Dictionary file not found")

    max_edit_distance_lookup = 2
    suggestion_verbosity = Verbosity.CLOSEST  # TOP, CLOSEST, ALL
    df_final = pd.DataFrame()
    for index , row in data.iterrows():
        # lookup suggestions for single-word input strings
        text = row[column]
        # max edit distance per lookup
        # (max_edit_distance_lookup <= max_edit_distance_dictionary)
        for input_term in text.split():
            suggestions = sym_spell.lookup(input_term, suggestion_verbosity,
                                       max_edit_distance_lookup)
            if len(suggestions)>0:
                df_local = pd.DataFrame({'Original Word':[input_term],'Replacement':[suggestions[0].term]})        
                df_final = df_final.append(df_local)
    return df_final

documents = [
   "This article is about the Golden State Warriors",
   "This article is about the Golden Arches",
   "This article is about state machines",
   "This article is about viking warriors"]

############ Text Preprocessing ############
processed_documents  = list(map(preprocess,documents))

#using list comprehension
# processed_documents  = [preprocess(x) for x in documents]
#using for loop
# processed_documents = []
# for x in documents:    
#     processed_documents.append(preprocess(x))

############ Spell Correction ############
df = pd.DataFrame(documents,columns=['documents'])
original_replacement_word = spelling_correction(df,'documents')

