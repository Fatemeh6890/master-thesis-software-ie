import os
import pandas as pd
import statistics
import json


def json_sort(json_file: str):
        file_number = int(json_file.split('_')[-1].split('.')[0])
        return file_number  


def get_list_of_json_files(path: str):
    json_files = [file for file in os.listdir(path) if file.endswith('.json')]
    json_files = sorted(json_files, key=json_sort)
    return json_files


def data(path: str, json_files: list):
    data = []
    for file in json_files:
        file_name = os.path.join(path, file)
        with open(file_name) as f:
            data_of_file = json.load(f)
            data.append(data_of_file)
    return data


def calculate_mean(list_of_data: list):
    list_of_f1_means = []
    list_of_precision_means = []
    list_of_recall_means = []
    for data in list_of_data:
        log = pd.DataFrame(data['log_history'])
        mean_f1_log = log['eval_overall_f1'].mean()
        mean_precision_log = log['eval_overall_precision'].mean()
        mean_recall_log = log['eval_overall_recall'].mean()
        list_of_f1_means.append(mean_f1_log)
        list_of_precision_means.append(mean_precision_log)
        list_of_recall_means.append(mean_recall_log)
    return statistics.mean(list_of_precision_means),  statistics.mean(list_of_recall_means), statistics.mean(list_of_f1_means)

        
def create_dictionary_write_in_text_file(name: str, precision: float, recall: float, f1: float):
    info_dict = {'training_bert_base_uncased': {'precision': precision,
                                                'recall': recall,
                                                'f1': f1}}
    with open("optimized/subtask2_optimized_results.txt", "a") as file:
        file.write("\n"+ json.dumps(info_dict) + "\n")