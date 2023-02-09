import argparse
import json
import random
import optuna

import utils.eval_utils

OUTPUT_ROOT = '../results/outputs'

def tune_and_eval(args):

    dev_path = f'{OUTPUT_ROOT}/{args.name}_dev.json'
    dev_test_path = f'{OUTPUT_ROOT}/{args.name}.json'
    dev_test_write_path = f'{OUTPUT_ROOT}/{args.name}_tuned.json'

    def objective(trial):
        offset = trial.suggest_float('offset', -10, 10)
        prec, recall, f1 = eval(dev_path, offset)
        return f1

    study = optuna.create_study(direction = 'maximize')
    study.optimize(objective, n_trials = 100)

    print('Optuna output:')
    print(study.best_params)

    prec, recall, f1 = eval(dev_path, 0)
    prec, recall, f1 = utils.eval_utils.format_prec_recall_f1(prec, recall, f1)
    print(f'Dev prec/recall/f1 (Before tunning): {prec} {recall} {f1}')

    prec, recall, f1 = eval(dev_path, study.best_params['offset'])
    prec, recall, f1 = utils.eval_utils.format_prec_recall_f1(prec, recall, f1)
    print(f'Dev prec/recall/f1: {prec} {recall} {f1}')
    
    prec, recall, f1 = eval(dev_test_path, 0, write_path = dev_test_write_path)
    prec, recall, f1 = utils.eval_utils.format_prec_recall_f1(prec, recall, f1)
    print(f'Dev test prec/recall/f1 (Before tuning): {prec} {recall} {f1}')

    prec, recall, f1 = eval(dev_test_path, study.best_params['offset'], write_path = dev_test_write_path)
    prec, recall, f1 = utils.eval_utils.format_prec_recall_f1(prec, recall, f1)
    print(f'Dev test prec/recall/f1: {prec} {recall} {f1}')

    return



def eval(path, offset, write_path = ''):

    with open(path, 'r') as f:
        data = json.load(f)

    n_pred, n_ref, n_hit = 0, 0, 0

    if write_path != '':
        out = []
    
    for line in data:

        logit = line['logit']
        label = line['label']

        if logit + offset >= 0 and label == 1:
            n_hit += 1
        if logit + offset >= 0:
            n_pred += 1
        if label == 1:
            n_ref += 1

        if write_path != '':
            if logit + offset >= 0:
                line['tuned_pred'] = 1
            else:
                line['tuned_pred'] = 0
            out.append(line)
    
    if write_path != '':
        print(f'Writing tuned output at {write_path}')
        with open(write_path, "w") as f:
            json.dump(out, f)
        
    prec, recall, f1 = utils.eval_utils.prec_recall_f1(n_pred, n_ref, n_hit)

    return prec, recall, f1

def weak_baseline(path, method='random'):
    
    with open(path, 'r') as f:
        data = json.load(f)

    n_pred, n_ref, n_hit = 0, 0, 0
    
    for line in data:

        label = line['label']

        if label == 1:
            n_ref += 1

        # Pred
        if method == 'majority':
            pred = True
        elif method == 'random':
            pred = random.random() < 0.289
        else:
            raise NotImplementedError
    
        if pred:
            n_pred += 1
        if pred and label == 1:
            n_hit += 1
        
    prec, recall, f1 = utils.eval_utils.prec_recall_f1(n_pred, n_ref, n_hit)
    
    return prec, recall, f1
    

if __name__ == '__main__':
    # path = '../results/outputs/albef_pool_vis.json'

    # prec, recall, f1 = weak_baseline(path, method='random')
    # prec, recall, f1 = utils.eval_utils.format_prec_recall_f1(prec, recall, f1)
    # print(f'Random prec/recall/f1: {prec} {recall} {f1}')

    # # It's actually minority
    # prec, recall, f1 = weak_baseline(path, method='majority')
    # prec, recall, f1 = utils.eval_utils.format_prec_recall_f1(prec, recall, f1)
    # print(f'majority prec/recall/f1: {prec} {recall} {f1}')
    # exit()

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--name', default='finetune_albef_mm_metadata_')
    parser.add_argument('--n_trials', default=100, type=int)
    
    args = parser.parse_args()

    tune_and_eval(args)
