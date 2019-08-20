from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re
 
cachedStopWords = stopwords.words("english")
 
def tokenize(text):
  min_length = 3
  words = map(lambda word: word.lower(), word_tokenize(text))
  words = [word for word in words if word not in cachedStopWords]
  tokens = (list(map(lambda token: PorterStemmer().stem(token),
                                   words)))
  p = re.compile('[a-zA-Z]+');
  filtered_tokens = list(filter (lambda token: p.match(token) and
                               len(token) >= min_length,
                               tokens))
  return filtered_tokens

from nltk.corpus import stopwords, reuters
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
stop_words = stopwords.words("english")


####### Classification #########

# List of document ids
documents = reuters.fileids()
 
train_docs_id = list(filter(lambda doc: doc.startswith("train"),
                            documents))
test_docs_id = list(filter(lambda doc: doc.startswith("test"),
                           documents))
 
train_docs = [reuters.raw(doc_id) for doc_id in train_docs_id]
test_docs = [reuters.raw(doc_id) for doc_id in test_docs_id]
 
# Tokenisation
vectorizer = TfidfVectorizer(stop_words=stop_words,
                             tokenizer=tokenize)
 
# Learn and transform train documents
vectorised_train_documents = vectorizer.fit_transform(train_docs)
vectorised_test_documents = vectorizer.transform(test_docs)
 
# Transform multilabel labels
mlb = MultiLabelBinarizer()
train_labels = mlb.fit_transform([reuters.categories(doc_id)
                                  for doc_id in train_docs_id])
test_labels = mlb.transform([reuters.categories(doc_id)
                             for doc_id in test_docs_id])
 
# Classifier
classifier = OneVsRestClassifier(LinearSVC(random_state=42))
classifier.fit(vectorised_train_documents, train_labels)
 
predictions = classifier.predict(vectorised_test_documents)

####### Evaluation #########
from sklearn.metrics import f1_score,precision_score,recall_score
 
precision = precision_score(test_labels, predictions,average='micro')
recall = recall_score(test_labels, predictions,average='micro')
f1 = f1_score(test_labels, predictions, average='micro')
 
print("Micro-average quality numbers")
print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))
 
precision = precision_score(test_labels, predictions,average='macro')
recall = recall_score(test_labels, predictions,average='macro')
f1 = f1_score(test_labels, predictions, average='macro')
 
print("Macro-average quality numbers")
print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))
