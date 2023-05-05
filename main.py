import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

sgd = SGD(learning_rate=0.01, momentum=0.9)

my_optimizer = SGD(learning_rate=0.01, momentum=0.0, nesterov=False)

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


nltk.download('punkt')
nltk.download('wordnet')

words = []
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json').read()
intents = json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenisieren jedes Wortes in dem Muster
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # Hinzufügen von Dokumenten
        documents.append((w, intent['tag']))
        # Hinzufügen zu den Klassenliste
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatisieren und Kleinbuchstaben für jedes Wort in der Wortliste
words = [WordNetLemmatizer().lemmatize(w.lower()) for w in words if w not in ignore_words]
# Sortieren und Entfernen von Duplikaten
words = sorted(list(set(words)))
# Sortieren und Entfernen von Duplikaten
classes = sorted(list(set(classes)))
# Dokumente erstellen
print(len(documents), "documents")
# Klassen erstellen
print(len(classes), "classes", classes)
# Wörter erstellen
print(len(words), "unique lemmatized words", words)

# Erstellen des Trainingsdatensatzes
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [WordNetLemmatizer().lemmatize(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# Mischen der Trainingsdaten und Erstellung von X und Y
np.random.shuffle(training)
training = np.array(training, dtype=object)
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Erstellen des Modells - 3 Layer
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Kompilieren des Modells
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Passen des Modells
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print("Modell erstellt")

with open('words.pkl', 'rb') as f:
    words = pickle.load(f)
with open('classes.pkl', 'rb') as f:
    classes = pickle.load(f)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Funktion zum Vorverarbeiten der Texte
def clean_up_sentence(sentence):
    # Tokenisieren des Benutzereingabesatzes
    sentence_words = nltk.word_tokenize(sentence)
    # Lemmatisieren jedes Wortes
    sentence_words = [WordNetLemmatizer().lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Funktion zum Erstellen des Eingabevektors
def bow(sentence, words, show_details=True):
    # Tokenisieren des Benutzereingabesatzes
    sentence_words = clean_up_sentence(sentence)
    # Erstellen eines Eingabevektors mit Nullen für jedes Wort
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # Erhöhen des Werts auf 1, wenn das Wort im Benutzereingabesatz vorkommt
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)

    return np.array(bag)

# Funktion zur Vorhersage der Klasse
def predict_class(sentence):
    # Erstellen des Eingabevektors
    p = bow(sentence, words, show_details=False)
    # Vorhersage der Klasse
    res = model.predict(np.array([p]))[0]
    # Filtern der Schwachstellen
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # Sortieren nach Wahrscheinlichkeit
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    return return_list

# Funktion zur Generierung von Antworten
def get_response(intents_list, intents_json):
    tag = intents_list[0][0]
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# Hauptfunktion zur Ausführung der Chatbot-Logik
def chatbot_response(text):
    intents_list = predict_class(text)
    response = get_response(intents_list, intents)
    return response

# Eingabeschleife
while True:
    message = input("You: ")
    print(chatbot_response(message))

with open('words.pkl', 'rb') as f:
    words = pickle.load(f)

with open('classes.pkl', 'rb') as f:
    classes = pickle.load(f)

model = Sequential.from_config(pickle.load(open('chatbot_model.h5_config.pkl', 'rb')))
model.load_weights('chatbot_model.h5')
