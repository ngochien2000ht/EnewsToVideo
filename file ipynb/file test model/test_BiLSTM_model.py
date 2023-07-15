import pickle
import re
from keras.models import load_model
import string
import numpy as np
from nltk import tokenize
from nltk.corpus import stopwords

re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')


def tokenizek(s): return re_tok.sub(r' \1 ', s).split()


model = load_model("../model_ner/model_ner.h5")
word2idx = pickle.load(open("../model_ner/word2Idx (1).pkl", "rb"))
tag2idx = pickle.load(open("../model_ner/idx2Label (1).pkl", "rb"))
index2tag = {idx: word for word, idx in tag2idx.items()}
tags = list(tag2idx.keys())

def down(sentences):

        re_tok = re.compile(f"([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])")
        sentence = re_tok.sub(r"  ", sentences).split()

        padded_sentence = sentence + [word2idx["--PADDING--"]] * (100 - len(sentence))
        padded_sentence = [word2idx.get(w, 0) for w in padded_sentence]

        pred = model.predict(np.array([padded_sentence]))
        pred = np.argmax(pred, axis=-1)
        k = []
        for w, p in zip(sentence, pred[0]):
            if index2tag[p] != "O":
                print(w," - ",index2tag[p])
down("Sau chiến tích vẻ vang tại Campuchia, VDV Nguyễn Thị Oanh được trao thưởng một chiếc xe oto Peugeot 2008")

def downimgbing(sentences):
    for i in range(len(sentences)):
        re_tok = re.compile(f"([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])")
        sentence = re_tok.sub(r"  ", sentences[i]).split()

        padded_sentence = sentence + [word2idx["--PADDING--"]] * (100 - len(sentence))
        padded_sentence = [word2idx.get(w, 0) for w in padded_sentence]

        pred = model.predict(np.array([padded_sentence]))
        pred = np.argmax(pred, axis=-1)
        k = []
        for w, p in zip(sentence, pred[0]):
            if index2tag[p] != "O":
                k.append(w)
        if k != []:
            keywords = " ".join(k)
            with open('readme2.txt', 'a', encoding="utf8") as f:
                f.write(keywords+"\n")
                f.close()

        else:
            pass

# doc = ''.join(open('temp/readme.txt', encoding="utf8").readlines())
# stop_words = set(stopwords.words('english'))
# total_sentences = tokenize.sent_tokenize(doc)
# downimgbing(total_sentences)
