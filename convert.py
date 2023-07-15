import os
import re
import shutil
import string
import numpy as np
from bing_image_downloader import downloader
from google_images_download import google_images_download
from nltk.corpus import stopwords
from nltk import tokenize
import underthesea
from keras.models import load_model
import pickle
import torch
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from transformers import BertTokenizerFast, BertForTokenClassification
from down_img_chrome import link_img, download_image

patterns = {
    '[àáảãạăắằẵặẳâầấậẫẩ]': 'a',
    '[đ]': 'd',
    '[èéẻẽẹêềếểễệ]': 'e',
    '[ìíỉĩị]': 'i',
    '[òóỏõọôồốổỗộơờớởỡợ]': 'o',
    '[ùúủũụưừứửữự]': 'u',
    '[ỳýỷỹỵ]': 'y'
}


# model = load_model("model_ner/model_ner.h5")
# word2idx = pickle.load(open("model_ner/word2Idx (1).pkl", "rb"))
# tag2idx = pickle.load(open("model_ner/idx2Label (1).pkl", "rb"))
# index2tag = {idx: word for word, idx in tag2idx.items()}


def convert1(text):
    output = text
    for regex, replace in patterns.items():
        output = re.sub(regex, replace, output)
        # deal with upper case
        output = re.sub(regex.upper(), replace.upper(), output)
    return output


def convert(text):
    textdown = []
    for i in text:
        output = i
        for regex, replace in patterns.items():
            output = re.sub(regex, replace, output)
            # deal with upper case
            output = re.sub(regex.upper(), replace.upper(), output)
        textdown.append(output)
    return textdown


# def downimg( sentences,topic_title): ### down img gg
#     for i in range(len(sentences)):
#         key = underthesea.ner(sentences[i])
#         k=[]
#         for t in key:
#             if t[3] != 'O':
#                 k.append(t[0])
#         if k != []:
#             keywords = " ".join(k)
#             response = google_images_download.googleimagesdownload()
#             keywords = convert1(str(topic_title)+" "+keywords)
#             # keywords = str(topic_title) + " " + keywords
#             arguments = {"keywords": keywords,
#                          "limit": 2,
#                          "size": "large",
#                          "output_directory": "temp/down",
#                          "image_directory": str(i),
#                          "print_urls": True}
#             response.download(arguments)
#             # print("sentence "+str(i)+": "+keywords)
#         else:
#             pass


# def downimgbing(sentences, topic_title):  ### down img bing
#     for i in range(len(sentences)):
#         key = underthesea.ner(sentences[i])
#         k = []
#         for t in key:
#             if t[3] != 'O':
#                 k.append(t[0])
#         if k != []:
#             keywords = " ".join(k)
#             # keywords = convert1(str(topic_title) + " " + keywords)
#             keywords = str(topic_title) + " " + keywords
#             downloader.download(keywords, limit=1, output_dir="temp/down/" + str(i),
#                                 adult_filter_off=True, force_replace=False, timeout=60)
#         else:
#             pass
#     for i in range(len(sentences)):
#         folder_video = "temp/down/" + str(i) + "/"
#         try:
#             os.path.exists(folder_video) == True
#             filevideo = os.listdir(folder_video)[0]
#             folder = "temp/down/" + str(i) + "/" + str(filevideo)
#             f0 = os.listdir(folder)[0]
#             # f1 = os.listdir(folder)[1]
#             shutil.copyfile(folder + "/" + f0, folder_video + f0)
#             # shutil.copyfile(folder + "/" + f1, folder_video + f1)
#             shutil.rmtree(folder)
#         except:
#             pass

# def downimgbing(sentences, topic_title): ### model bi-LSTM
#     for i in range(len(sentences)):
#         # key = underthesea.ner(sentences[i])
#         re_tok = re.compile(f"([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])")
#         sentence = re_tok.sub(r"  ", sentences[i]).split()
#
#         padded_sentence = sentence + [word2idx["--PADDING--"]] * (100 - len(sentence))
#         padded_sentence = [word2idx.get(w, 0) for w in padded_sentence]
#
#         pred = model.predict(np.array([padded_sentence]))
#         pred = np.argmax(pred, axis=-1)
#         k = []
#         for w, p in zip(sentence, pred[0]):
#             if index2tag[p] != "O":
#                 k.append(w)
#             if index2tag[p] != "O":
#                 k.append(index2tag[p])
#         if k != []:
#             keywords = " ".join(k)
#             keywords = str(topic_title) + " " + keywords
#             downloader.download(keywords, limit=1, output_dir="temp/down/" + str(i),
#                                 adult_filter_off=True, force_replace=False, timeout=60)
#
#         else:
#             pass
#     for i in range(len(sentences)):
#         folder_video = "temp/down/" + str(i) + "/"
#         try:
#             os.path.exists(folder_video) == True
#             filevideo = os.listdir(folder_video)[0]
#             folder = "temp/down/" + str(i) + "/" + str(filevideo)
#             f0 = os.listdir(folder)[0]
#             shutil.copyfile(folder + "/" + f0, folder_video + f0)
#             shutil.rmtree(folder)
#         except:
#             pass

# def down_imgs_from_keywords(topic_title):
#     doc = ''.join(open('temp/readme.txt', encoding="utf8").readlines())
#     stop_words = set(stopwords.words('english'))
#     total_sentences = tokenize.sent_tokenize(doc)
#     downimgbing(total_sentences, topic_title)
#     # downimg(total_sentences,topic_title)
#
#     return total_sentences

unique_labels = {'I-ORGANIZATION', 'I-PERSONTYPE', 'B-ORGANIZATION', 'I-LOCATION', 'O', 'B-PERSON', 'B-LOCATION',
                 'I-PRODUCT', 'I-PERSON', 'B-PRODUCT', 'B-PERSONTYPE'}
ids_to_labels = {v: k for v, k in enumerate(sorted(unique_labels))}


def run_model():
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')

    class BertModel(torch.nn.Module):

        def __init__(self):
            super(BertModel, self).__init__()
            self.bert = BertForTokenClassification.from_pretrained('bert-base-multilingual-cased',
                                                                   num_labels=len(unique_labels))

        def forward(self, input_id, mask, label):
            output = self.bert(input_ids=input_id, attention_mask=mask, labels=label, return_dict=False)

            return output

    model = BertModel()
    try:
        model.load_state_dict(torch.load('model_ner/bert_model.pth'))
        print("gpu")
    except:
        model.load_state_dict(torch.load('model_ner/bert_model.pth', map_location=torch.device('cpu')))
    print('cpu')

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        model = model.cuda()
    return device, model, tokenizer


def align_word_ids(texts, tokenizer):
    tokenized_inputs = tokenizer(texts, padding='max_length', max_length=512, truncation=True)
    word_ids = tokenized_inputs.word_ids()
    previous_word_idx = None
    label_ids = []
    label_all_tokens = False
    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)
        elif word_idx != previous_word_idx:
            try:
                label_ids.append(1)
            except:
                label_ids.append(-100)
        else:
            try:
                label_ids.append(1 if label_all_tokens else -100)
            except:
                label_ids.append(-100)
        previous_word_idx = word_idx
    return label_ids


def evaluate_one_text(model, sentence, tokenizer, device):
    sentence = re.sub(r'[^\w\s]', '', sentence)
    text = tokenizer(sentence, padding='max_length', max_length=512, truncation=True, return_tensors="pt")

    mask = text['attention_mask'].to(device)
    input_id = text['input_ids'].to(device)
    label_ids = torch.Tensor(align_word_ids(sentence, tokenizer)).unsqueeze(0).to(device)

    logits = model(input_id, mask, None)
    logits_clean = logits[0][label_ids != -100]

    predictions = logits_clean.argmax(dim=1).tolist()
    prediction_label = [ids_to_labels[i] for i in predictions]
    word_sentence = sentence.split()
    k = []
    for i in range(len(prediction_label)):
        if prediction_label[i] != 'O':
            k.append(word_sentence[i])
    key = ' '.join(k)
    return key


def downimg(total_sentences, topic_title, Win, device, model, tokenizer):
    for i in range(len(total_sentences)):
        try:
            k = evaluate_one_text(model, total_sentences[i], tokenizer, device)
            if k != str():
                stringl = topic_title + " " + k
                link = link_img(stringl, Win)
                download_image("temp/down/", i, link)
        except Exception as exc:
            print(exc)


def down_imgs_from_keywords(topic_title, device, model, tokenizer):
    doc = ''.join(open('temp/readme.txt', encoding="utf8").readlines())
    stop_words = set(stopwords.words('english'))
    total_sentences = tokenize.sent_tokenize(doc)
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    PATH = "chromedriver_win32/chromedriver.exe"
    wd = webdriver.Chrome(PATH, options=chrome_options)
    downimg(total_sentences, topic_title, wd, device, model, tokenizer)
    wd.quit()

    return total_sentences
