import torch
from transformers import BertTokenizerFast, BertForTokenClassification
import re

unique_labels = {'I-ORGANIZATION', 'I-PERSONTYPE', 'B-ORGANIZATION', 'I-LOCATION', 'O', 'B-PERSON', 'B-LOCATION',
                 'I-PRODUCT', 'I-PERSON', 'B-PRODUCT', 'B-PERSONTYPE'}
ids_to_labels = {v: k for v, k in enumerate(sorted(unique_labels))}

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

label_all_tokens = False

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
if use_cuda:
    model = model.cuda()


def align_word_ids(texts):
    tokenized_inputs = tokenizer(texts, padding='max_length', max_length=512, truncation=True)
    word_ids = tokenized_inputs.word_ids()
    previous_word_idx = None
    label_ids = []
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

def evaluate_one_text(model, sentence):


    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    text = tokenizer(sentence, padding='max_length', max_length = 512, truncation=True, return_tensors="pt")

    mask = text['attention_mask'].to(device)
    input_id = text['input_ids'].to(device)
    label_ids = torch.Tensor(align_word_ids(sentence)).unsqueeze(0).to(device)

    logits = model(input_id, mask, None)
    logits_clean = logits[0][label_ids != -100]

    predictions = logits_clean.argmax(dim=1).tolist()
    prediction_label = [ids_to_labels[i] for i in predictions]
    print(sentence)
    print(prediction_label)

evaluate_one_text(model, 'Sau chiến tích vẻ vang tại Campuchia, VDV Nguyễn Thị Oanh được trao thưởng một chiếc xe oto Peugeot 2008')



# def evaluate_one_text(model, sentence):
#     sentence = re.sub(r'[^\w\s]', '', sentence)
#     text = tokenizer(sentence, padding='max_length', max_length=512, truncation=True, return_tensors="pt")
#
#     mask = text['attention_mask'].to(device)
#     input_id = text['input_ids'].to(device)
#     label_ids = torch.Tensor(align_word_ids(sentence)).unsqueeze(0).to(device)
#
#     logits = model(input_id, mask, None)
#     logits_clean = logits[0][label_ids != -100]
#
#     predictions = logits_clean.argmax(dim=1).tolist()
#     prediction_label = [ids_to_labels[i] for i in predictions]
#     word_sentence = sentence.split()
#     k = []
#     key = str()
#     for i in range(len(prediction_label)):
#         if prediction_label[i] != 'O':
#             k.append(word_sentence[i])
#     key = ' '.join(k)
#     return key
#
#
# def make_kw(key):
#     keywords = key
#     with open('readme2.txt', 'a', encoding="utf8") as f:
#         f.write(keywords + "\n")
#         f.close()
#
#
# doc = ''.join(open('temp/readme.txt', encoding="utf8").readlines())
# with open('temp/readme.txt', "r", encoding="utf8") as f:
#     for line in f:
#         sentence = line.strip()
#         try:
#             key = evaluate_one_text(model, sentence)
#             make_kw(key)
#         except:
#             print(1, line.strip())
