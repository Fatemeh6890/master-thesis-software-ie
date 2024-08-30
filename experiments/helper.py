import evaluate
import numpy as np
import torch
import pandas as pd
from transformers import BertTokenizerFast, AutoTokenizer


metric = evaluate.load("seqeval")

def ner_eval_metrics(eval_preds, id_to_label):
    logits, labels = eval_preds
    # becase the logics and probabilities both are in the same order, we don't need to aply softmax here
    predictions = np.argmax(logits, axis=-1)
    # now we need to remove all the values, where the label is -100
    # before passing to metric.compute we should have these inputs as a list
    true_labels = [[id_to_label[l] for l in label if l != -100]
                   for label in labels]
    true_predictions = [[id_to_label[p] for p,l in zip(prediction, label) if l != -100]
                        for prediction, label in zip(predictions, labels)]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels, zero_division=0)
    return all_metrics

def align_labels(word_ids, predicted_label_ids):
    """map predicted labels from BERT-tokens to words"""
    current_word_id = None
    output_label_ids = []

    for index, word_id in enumerate(word_ids):
        if word_id is None:
            # filter out None at beginning and end
            continue
        elif word_id != current_word_id:
            # a new word beginn
            # if current word_idx is != prev add the corresponding token, its the most regular case
            label = predicted_label_ids[index]
            output_label_ids.append(label)

        else:
            # for subtekens
            continue
        current_word_id = word_id
    return output_label_ids


def compute_metrics(eval_preds, class_labels):
    logits, labels = eval_preds
    # becase the logics and probabilities both are in the same order, we don't need to aply softmax here
    predictions = np.argmax(logits, axis=-1)
    # now we need to remove all the values, where the label is -100
    # before passing to metric.compute we should have these inputs as a list
    true_labels = [[class_labels[l] for l in label if l != -100] for label in labels]

    true_predictions = [
        [class_labels[p] for p, l in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return all_metrics


def convert_ids_to_labels(sentence, model, tokenizer, id_to_label):
    sentence = sentence.split()
    inputs = tokenizer(
        [sentence],
        truncation=True,
        is_split_into_words=True,
        padding=True,
        return_tensors="pt",
    )
    word_ids = inputs.word_ids()
    with torch.no_grad():
        model.eval()
        outputs = model(**inputs)
    prediction = outputs.logits.argmax(dim=2)
    prediction = prediction[0].tolist()
    predictions_for_words = align_labels(word_ids, prediction)
    predicted_labels = [id_to_label[id] for id in predictions_for_words]
    return labels

def map_labels(tokenized_texts, labels):
    """
    
    """
    labels_mapped = []
    for idx in range(len(labels)):
        word_ids_sample = tokenized_texts.word_ids(idx)
        labels_sample = labels[idx]
        labels_sample_mapped = get_mapped_labels(word_ids_sample, labels_sample)
        labels_mapped.append(labels_sample_mapped)
    tokenized_texts["labels"] = labels_mapped

def tokenized_align_labels(tokenized_train_inputs, input_label_ids, label_all_tokens=False):
     ## The below function 'tokenize_and_align_labels' does 2 jobs
    #  1. set -100 as the label for these special tokens
    #  2. mask the subword representation after the first subword
    ## Then we align the labels with the tken ids using the strategy we picked
    # map labels from words to BERT-Tokens
    """
    The label_all_tokens parameter is often used when tokenizing and encoding text for token classification.
    - When label_all_tokens=True, it indicates that the model should assign labels to all tokens generated during
      tokenization, including subtokens.
    - When label_all_tokens=False (or not specified), only the label for the first token of each word (or the special tokens)
      is considered during training.

    """
    word_ids = tokenized_train_inputs.word_ids()
    output_label_ids = get_mapped_labels(word_ids, input_label_ids, label_all_tokens)
    assert len(tokenized_train_inputs["input_ids"][0]) == len(
        output_label_ids
    )  # output does not have the same length as bert-tokenized_text
    output_label_ids = torch.tensor(output_label_ids)
    output_label_ids = output_label_ids.unsqueeze(0)
    tokenized_train_inputs["labels"] = output_label_ids
    return tokenized_train_inputs


def get_mapped_labels(word_ids, input_label_ids, label_all_tokens=False):
    current_word_id = None
    output_label_ids = []
    for word_id in word_ids:
        if word_id is None:
            # None at beginning and end of the sequence
            # set -100 as the label for these special tokens
            output_label_ids.append(-100)
        elif word_id != current_word_id:
            # a new word beginn
            # if current word_idx is != current,add the corresponding token (its the most regular case)
            label = input_label_ids[word_id]
            output_label_ids.append(label)
        else:
            # for non-first subwort-tokens
            if label_all_tokens:
                label = input_label_ids[word_id]
                label = torch.tensor(label)
                output_label_ids.append(label)
            else:
                output_label_ids.append(-100)
        current_word_id = word_id
    return output_label_ids

def reduce_to_entity_type_labels(labels):
    new_labels = []
    for label in labels:
        bio_ent = label.split("_")[0]
        new_labels.append(bio_ent)
    return new_labels


def reduce_to_intention_type_labels(labels):
    new_labels = []
    for label in labels:
        if label == "O":
            new_labels.append(label)
        else:
            bio = label[0]
            intention = label.split("_")[1]
            bio_intention = f"{bio}-{intention}"  # e.g. 'B-creation'
            new_labels.append(bio_intention)
    return new_labels

def convert_into_dataframe(data):
    sentences = [eval(data[i]) for i in range(0, len(data), 3)]
    labels = [eval(data[i]) for i in range(1, len(data), 3)]
    ids = [eval(data[i]) for i in range(2, len(data), 3)]

    # Create a DataFrame
    dataframe = pd.DataFrame({'sentences': sentences, 'labels': labels, 'ids': ids})
    return dataframe

def tokenization(input_data):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    tokenized_train_inputs = tokenizer(
        input_data,
        return_tensors="pt",
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=512,
    )
    
    return tokenized_train_inputs

def class_labels(txt_file='../../data/subtask1/subtask1_train.labels.txt'):
    # Open the label data
    with open(txt_file, 'r', encoding='utf-8') as file:
        labels = file.readlines()
    
    class_labels = []
    for line in labels:
        labels_list = line.split()
        for label in labels_list:
            if label not in class_labels:
                class_labels.append(label)
    return class_labels



def get_entities(token, entity_bio_tags):
    entity_infos = []
    for pos_idx, bio_tag in enumerate(entity_bio_tags):
        bio = bio_tag[0]
        label_all = bio_tag[2:]
        label = label_all
        intention = None
        if "_" in label_all: # intention exist
            label, intention = label_all.split("_")
        if bio == "B":
            ent = dict(
                text=token[pos_idx],
                label=label,
                intention=intention,
                begin=pos_idx,
                end=pos_idx
            )
            entity_infos.append(ent)
        if bio == "I": # Update last Entity
            text_additional = token[pos_idx]
            ent_last = entity_infos[-1]
            ent_last['text'] += f" {text_additional}"
            ent_last['end'] = pos_idx
    return entity_infos



def get_relations(relation_line):
    sentence_relations = []  # List to store information for the current line
    if relation_line.strip():  # Check if the line is not empty or whitespace
        relations_info_list = relation_line.split(';;')
        for relation_subject_object in relations_info_list:
            rel, sub, obj = relation_subject_object.split('\t')
            sentence_relations.append({'subject': int(sub), 'relation_type': rel, 'object': int(obj)})
    return sentence_relations


def sentence_allowed_subj_obj(sent, allow_inverse):
    for rel in sent['relations']:
        subj = sent['entities'][rel['subject']]
        subj_type = subj['label']
        obj = sent['entities'][rel['object']]
        obj_type = obj['label']
        yield subj_type, obj_type
        if allow_inverse:
            yield obj_type, subj_type
            
       
    
def build_relation_reprentation(sent, subj, obj, rel):
    representation = build_sentence_subj_obj(sent, subj, obj)
    return representation, rel


def build_sentence_subj_obj(sent, subj, obj):
     sub_obj = f"[{subj['label']}: '{subj['text']}'], [{obj['label']}: '{obj['text']}']"
     return f"{sent['sentence']} [SEP] {sub_obj}"
    
