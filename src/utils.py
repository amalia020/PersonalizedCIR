"""
# NOTE: This was missing, added parts from:
# https://github.com/fengranMark/ConvRelExpand/blob/main/scripts/utils.py
Most code is our addition to properly extract duplicated code,
or to make hardcoded parts dynamic.
"""
import json
import os
from collections import defaultdict

import logging
import numpy as np
import random
import re
import pytrec_eval
import torch

        
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_processed_sample_ids(output_path):
    """
    Loads sample id's to check if there are sample-id's left to process
    Useful for: 
    (1) Check if there are no samples skipped
    (2) avoid reprocessing if job / openAI interrupts
    """
    processed_sample_ids = set()
    if os.path.exists(output_path):
        with open(output_path, 'r') as outfile:
            for line in outfile:
                try:
                    data = json.loads(line)
                    sample_id = data.get('sample_id')
                    if sample_id:
                        processed_sample_ids.add(sample_id)
                except json.JSONDecodeError:
                    print("Skipping")
    return processed_sample_ids


def get_assessed_turn_ids(path:str="data/2023-qrels.all-turns.txt"):
    """
    Get the ids of the turns that have been assessed
    We checked this to be the same as the hardcoded list they used originally.
    """
    idxs = set()
    with open(path, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 4:
                idxs.add(parts[0].replace("_", "-"))
    return idxs


def has_used_provenance(ptkb_provenance, provenance_seen, number):
    for item in ptkb_provenance:
        if item in provenance_seen[number]:
            return True  
    return False 


def get_relevant_assessed_turn_ids(set176, new_ptkb=True, file="data/2023_test_topics_flattened.jsonl"):
    data_list = []
    relevant_entries = []
    provenance_seen = {}
    with open(file) as f:
        for line in f:
            data_list.append(json.loads(line))

    if data_list[0].get('sample_id') != '9-1-1':
        raise ValueError("Data list is not sorted. List should start with sample_id '9-1-1'.")
    
    for data in data_list:
        # number is converstation like 9-1
        number = data.get('number', '')
        # sample_id is turn in converstation like 9-1-1 is turn 1 of conversation
        sample_id = data.get('sample_id', '')
        ptkb_provenance = data.get('ptkb_provenance', [])
        
        # new_ptkb for modified methodology
        if new_ptkb:
            # keep history of used provenance for each converstation
            if number not in provenance_seen:
                provenance_seen[number] = set()
            # check if any provenance_ptkb item in current turn was already used in converstation
            if has_used_provenance(ptkb_provenance, provenance_seen, number):
                continue
            provenance_seen[number].update(ptkb_provenance)
        if sample_id not in set176:
            continue
        # Always filter out entries with empty ptkb_provenance list
        if not ptkb_provenance:
            continue
        relevant_entries.append(data['sample_id'])
    print(f'found {len(relevant_entries)} turns')
    return relevant_entries


def parse_relevant_ids(args, verbose=True):
    relevant_ids = get_assessed_turn_ids()  
    if args.subset:
        if verbose:
            print("Filtering empty provenance is ON")
            print(f"Corrected methodology filtering is {'ON' if args.new_ptkb else 'OFF'}")
        relevant_ids = get_relevant_assessed_turn_ids(relevant_ids, new_ptkb=args.new_ptkb, file=args.relevant_ptkb_path)
    elif verbose:
        print("No filter applied to assessed 176 entries")
    return relevant_ids


def is_relevant(qid, relevant_ids):
    if qid in relevant_ids:
        return True
    
    if len(qid.split("-")) == 4:
        return qid[:qid.rfind("-")] in relevant_ids
    return False


def load_provenance_dict(file_path, type='LLM'):
    provenance_dict = {}
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            sample_id = data.get('sample_id', '')
            if type == 'LLM' or type == 'machine':
                select = data.get(f'{type}_select', '')
            elif type == 'None':
                select = []
            elif type == 'All':
                select = list(data.get(f'ptkb', '').keys())

            provenance_dict[sample_id] = select
    return provenance_dict


def demonstrate(shot):
    if shot == 1:
        demo = ["1-1"]
    elif shot == 3:
        demo = ["1-1","1-2","2-1"]
    elif shot == 5:
        demo = ["1-1","1-2","2-1","2-2","7-1"]
    demo_text = ''
    with open('data/2023_train_topics.json', 'r') as file:
        data = json.load(file)

        i = 0
        for entry in data:
            if entry['number'] in demo:
                turns = entry["turns"]
                demo_question = [i["utterance"] for i in turns]
                demo_rewrite = [i["resolved_utterance"] for i in turns]
                demo_response = [i["response"] for i in turns]
                demo_ptkb_prov = [i["ptkb_provenance"] for i in turns]
                qra = ''
                for q,rw,rp,prov in zip(demo_question,demo_rewrite,demo_response,demo_ptkb_prov):
                    qra += 'Question: ' + q + '\n' + 'provenance: ' + str(prov) + '\n' + 'Rewrite: ' + rw + '\n' + 'Response: ' + rp + '\n\n'
                i += 1
                demo_text += '# Example ' + str(i) + '\n\n' + 'User\'s information:' + str(entry["ptkb"]) + '\n\n' + qra
    return demo_text

def extract_numbers_from_dict(input_dict):
    output_dict = {}
    for key, value in input_dict.items():
        if isinstance(value, list):
            value_str = ' '.join(map(str, value))
        else:
            # Ensure the value is a string
            value_str = str(value)
        # Use regex to find all numbers in the value
        numbers = re.findall(r'\d+', value_str)
        output_dict[key] = numbers
    return output_dict


def get_avg_eval_results(res1, res2 = None):
    map_list = [v['map'] for v in res1.values()]
    mrr_list = [v['recip_rank'] for v in res1.values()]
    recall_20_list = [v['recall_20'] for v in res1.values()]
    recall_1000_list = [v['recall_1000'] for v in res1.values()]
    precision_20_list = [v['P_20'] for v in res1.values()]

    res2 = res2 if res2 else res1
    ndcg_3_list = [v['ndcg_cut_3'] for v in res2.values()]
    ndcg_5_list = [v['ndcg_cut_5'] for v in res2.values()]
    ndcg_1000_list = [v['ndcg_cut_1000'] for v in res2.values()]

    res = {
            "MRR": float(round(np.average(mrr_list)*100, 5)),
            "NDCG@3": float(round(np.average(ndcg_3_list)*100, 5)),
            "NDCG@5": float(round(np.average(ndcg_5_list)*100, 5)),
            "NDCG@1000": float(round(np.average(ndcg_1000_list)*100, 5)),
            "Precision@20": float(round(np.average(precision_20_list)*100, 5)),
            "Recall@20": float(round(np.average(recall_20_list)*100, 5)),
            "Recall@1000": float(round(np.average(recall_1000_list)*100, 5)),
            "MAP": float(round(np.average(map_list)*100, 5)),
    }
    return res

def calculate_trec_res(run_file, qrel_file, rel_threshold, automatic_method=False):
    with open(qrel_file, 'r') as f:
        qrel_data = f.readlines()
    
    qrels = defaultdict(dict)
    qrels_ndcg = defaultdict(dict)
    
    # load from ground truth
    for line in qrel_data:
        line = line.strip().split()
        query = line[0].replace('_', '-')
        passage = line[2]
        rel = int(line[3])

        # for NDCG
        qrels_ndcg[query][passage] = rel
        # for MAP, MRR, Recall
        qrels[query][passage] = int(rel >= rel_threshold)

    # the PyPI version of PyTrecEval is outdated and contains known (fixed) bugs.
    # So we have to reconstruct it everytime to make metrics work properly.
    eval_general = lambda: pytrec_eval.RelevanceEvaluator(qrels, {"map", "recip_rank", "recall.5", "recall.10", "recall.20", "recall.1000","P_20"})
    eval_ndcg = lambda: pytrec_eval.RelevanceEvaluator(qrels_ndcg, {"ndcg_cut.3", "ndcg_cut.5", "ndcg_cut.1000"})

    with open(run_file, 'r' )as f:
        run_data = f.readlines()

    if automatic_method:
        run_per_ptkb = defaultdict(lambda: defaultdict(dict))
        for line in run_data:
            line = line.split()
            query = line[0]
            ptkb = query.split("-")[-1]
            qid = query[:query.rfind("-")] 
            pktb_run = run_per_ptkb[ptkb]
            passage = line[2]
            rel = int(line[4])
            pktb_run[qid][passage] = rel

        result_dict = defaultdict(lambda: defaultdict(dict))
        for pktb, run in run_per_ptkb.items():
            res = eval_general().evaluate(run)
            for qid, metric_dict in res.items():
                result_dict[qid][pktb].update(metric_dict)

            res = eval_ndcg().evaluate(run)
            for qid, metric_dict in res.items():
                result_dict[qid][pktb].update(metric_dict)
        best_result_dict = {}
        for qid, pktb_dict in result_dict.items():
            # for each query, take the one with the highest average metric (so sum) across all pktb
            metric_dict = max(pktb_dict.values(), key=lambda d: sum(d.values()))
            best_result_dict[qid] = metric_dict
        total_result = get_avg_eval_results(best_result_dict)

    else:
        runs = defaultdict(dict)
        for line in run_data:
            line = line.split()
            query = line[0]
            passage = line[2]
            rel = float(line[4])
            runs[query][passage] = rel

        res1 = eval_general().evaluate(runs)
        res2 = eval_ndcg().evaluate(runs)
        total_result = get_avg_eval_results(res1, res2)
        
    logging.info("---------------------Evaluation results:---------------------")
    logging.info(total_result)
    return total_result


def calculate_trec_res_NDCG(run_file, qrel_file, rel_threshold, automatic_method=False):
    with open(qrel_file, 'r') as f:
        qrel_data = f.readlines()
    
    qrels = defaultdict(dict)
    qrels_ndcg = defaultdict(dict)
    
    # Load ground truth relevance judgments
    for line in qrel_data:
        line = line.strip().split()
        query = line[0].replace('_', '-')
        passage = line[2]
        rel = int(line[3])

        # For NDCG
        qrels_ndcg[query][passage] = rel
        # For MAP, MRR, Recall
        qrels[query][passage] = int(rel >= rel_threshold)

    eval_general = lambda: pytrec_eval.RelevanceEvaluator(qrels, {"map", "recip_rank", "recall.5", "recall.10", "recall.20", "recall.1000", "P_20"})
    eval_ndcg = lambda: pytrec_eval.RelevanceEvaluator(qrels_ndcg, {"ndcg_cut.3", "ndcg_cut.5", "ndcg_cut.1000"})

    with open(run_file, 'r') as f:
        run_data = f.readlines()

    if automatic_method:
        run_per_ptkb = defaultdict(lambda: defaultdict(dict))
        for line in run_data:
            line = line.split()
            query = line[0]
            ptkb = query.split("-")[-1]
            qid = query[:query.rfind("-")]
            pktb_run = run_per_ptkb[ptkb]
            passage = line[2]
            rel = int(line[4])
            pktb_run[qid][passage] = rel

        # Evaluate runs
        result_dict = defaultdict(lambda: defaultdict(dict))
        for ptkb, run in run_per_ptkb.items():
            # Compute general metrics
            res_general = eval_general().evaluate(run)
            for qid, metrics in res_general.items():
                result_dict[qid][ptkb].update(metrics)
            # Compute NDCG metrics
            res_ndcg = eval_ndcg().evaluate(run)
            for qid, metrics in res_ndcg.items():
                result_dict[qid][ptkb].update(metrics)

        best_result_dict_baseline = {}
        best_result_dict_always_ptkb = {}

        for qid, pktb_dict in result_dict.items():
            # method 1: Using "no ptkb" (index 0) can be selected as it can give the highest NDCG@3
            best_metric_baseline = max(
                pktb_dict.values(),
                key=lambda metrics: metrics.get("ndcg_cut_3", 0)
            )
            best_result_dict_baseline[qid] = best_metric_baseline

            # method 2: always select a ptkb even if it's worse than using "no ptkb"
            always_ptkb = {ptkb: metrics for ptkb, metrics in pktb_dict.items() if ptkb != "0"}
            best_metric_always_ptkb = max(
                always_ptkb.values(),
                key=lambda metrics: metrics.get("ndcg_cut_3", 0)
            )
            best_result_dict_always_ptkb[qid] = best_metric_always_ptkb

        total_result_baseline = get_avg_eval_results(best_result_dict_baseline)
        total_result_always_ptkb = get_avg_eval_results(best_result_dict_always_ptkb)

        logging.info("Possibly no ptkb results: %s", total_result_baseline)
        logging.info("Always a ptkb query results: %s", total_result_always_ptkb)

        return total_result_baseline, total_result_always_ptkb

    else:
        # Evaluate without automatic method
        runs = defaultdict(dict)
        for line in run_data:
            line = line.split()
            query = line[0]
            passage = line[2]
            rel = int(line[4])
            runs[query][passage] = rel

        res1 = eval_general().evaluate(runs)
        res2 = eval_ndcg().evaluate(runs)
        total_result = get_avg_eval_results(res1, res2)

        logging.info("---------------------Evaluation results:---------------------")
        logging.info(total_result)
        return total_result


