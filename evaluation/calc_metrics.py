from sklearn.metrics import precision_recall_fscore_support
import evaluate
import pandas as pd

metric = evaluate.load('seqeval')

def compute(predictions, references):
    performance = metric.compute(predictions=predictions, references=references)
    micro = pd.Series({k[8:]: v for k, v in performance.items() if k.startswith("overall_")})
    label_performance = {k: v for k, v in performance.items() if not k.startswith("overall_")}
    
    metrics_df = pd.DataFrame(label_performance).T
    weights = metrics_df.number.divide(metrics_df.number.sum())
    weighted_average_macro = metrics_df[["precision", "recall", "f1"]].multiply(weights, axis=0).sum()
    metrics_df.loc["micro"] = micro
    metrics_df.loc["macro"] = metrics_df[["precision", "recall", "f1"]].mean()
    metrics_df.loc["macro_weighted"] = weighted_average_macro
    return metrics_df


def load_bio(fn):
    return [line.split() for line in open(fn).read().split("\n")]


def compute_metrics(eval_preds, label_names):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    labels = [label_names[l] for l in labels]
    predictions = pd.Series([label_names[p] for p in predictions])
    n_predicted = predictions.value_counts().rename("n_predicted")
    metrics = precision_recall_fscore_support(labels, predictions, labels=label_names, zero_division=0)
    metrics = pd.DataFrame(metrics, index=["prec", "recall", "f1", "support"], columns=label_names).T
    f = metrics.index != "nil"
    metrics = metrics[f].copy()
    ## weighted f1
    weights = metrics.support / metrics.support.sum()
    metrics = metrics.join(n_predicted)
    metrics["n_predicted"] = metrics.n_predicted.fillna(0)
    metric_info = dict(
        eval_f1_macro_weighted = (metrics.f1 * weights).sum(),
        eval_support = metrics.support.sum(),
        eval_n_predicted = metrics.n_predicted.fillna(0).sum()
    )
    for label, row in metrics.iterrows():
        for metr, value in row.to_dict().items():
            metric_info[f"eval_{metr}_{label}"] = value
    return metric_info

def current_compute_metrics(predictions, labels, label_names):
    predictions = pd.Series([label_names[p] for p in predictions])
    labels = pd.Series([label_names[l] for l in labels])
    
    n_predicted = predictions.value_counts().rename("n_predicted")
    metrics = precision_recall_fscore_support(labels, predictions, labels=label_names, zero_division=0)
    metrics = pd.DataFrame(metrics, index=["prec", "recall", "f1", "support"], columns=label_names).T
    
    f = metrics.index != "nil"
    metrics = metrics[f].copy()
    
    # Weighted F1
    weights = metrics.support / metrics.support.sum()
    metrics = metrics.join(n_predicted)
    metrics["n_predicted"] = metrics.n_predicted.fillna(0)
    
    metric_info = dict(
        eval_f1_macro_weighted=(metrics.f1 * weights).sum(),
        eval_support=metrics.support.sum(),
        eval_n_predicted=metrics.n_predicted.fillna(0).sum()
    )
    
    for label, row in metrics.iterrows():
        for metr, value in row.to_dict().items():
            metric_info[f"eval_{metr}_{label}"] = value
    
    return metric_info


def load_relation_info(fn):
    with open(fn, 'r') as f:
        data = f.readlines()
    return data


def parse_content(content):
    parsed_data = {}
    for idx, line in enumerate(content): 
        line = line.strip()
        if line:  # Ignore empty lines
            predictions = line.split(";")
            for pred in predictions:
                relation, entity1, entity2 = pred.split("\t")
                parsed_data[(idx, int(entity1), int(entity2))] = relation
    return parsed_data