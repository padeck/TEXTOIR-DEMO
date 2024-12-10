
import os
import torch
import numpy as np
import pandas as pd
from pytorch_pretrained_bert.modeling import WEIGHTS_NAME, CONFIG_NAME

def save_npy(npy_file, path, file_name):
    npy_path = os.path.join(path, file_name)
    np.save(npy_path, npy_file)

def load_npy(path, file_name):
    npy_path = os.path.join(path, file_name)
    npy_file = np.load(npy_path)
    return npy_file

def save_model(model, model_dir):

    save_model = model.module if hasattr(model, 'module') else model  
    model_file = os.path.join(model_dir, WEIGHTS_NAME)
    model_config_file = os.path.join(model_dir, CONFIG_NAME)
    torch.save(save_model.state_dict(), model_file)
    with open(model_config_file, "w") as f:
        f.write(save_model.config.to_json_string())

def restore_model(model, model_dir):
    output_model_file = os.path.join(model_dir, WEIGHTS_NAME)
    model.load_state_dict(torch.load(output_model_file))
    return model

def save_results(args, test_results):
    # Save y_pred and y_true to files
    pred_labels_path = os.path.join(args.method_output_dir, 'y_pred.npy')
    np.save(pred_labels_path, test_results['y_pred'])
    true_labels_path = os.path.join(args.method_output_dir, 'y_true.npy')
    np.save(true_labels_path, test_results['y_true'])

    # Create a copy to modify for saving other results
    results_copy = test_results.copy()

    # Remove unnecessary keys from the copy
    if 'y_true' in results_copy.keys():
        del results_copy['y_true']
    if 'y_prob' in results_copy.keys():
        del results_copy['y_prob']
    if 'y_pred' in results_copy.keys():
        del results_copy['y_pred']
    if 'y_feat' in results_copy.keys():
        del results_copy['y_feat']

    # Ensure results directory exists
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # Save metadata and results to a CSV file
    import datetime
    created_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    metadata = {
        'dataset': args.dataset,
        'method': args.method,
        'backbone': args.backbone,
        'known_cls_ratio': args.known_cls_ratio,
        'labeled_ratio': args.labeled_ratio,
        'loss': args.loss_fct,
        'seed': args.seed,
        'num_train_epochs': args.num_train_epochs,
        'created_time': created_time,
    }

    results = dict(results_copy, **metadata)
    results_path = os.path.join(args.result_dir, args.results_file_name)

    # Write results to CSV
    if not os.path.exists(results_path) or os.path.getsize(results_path) == 0:
        df = pd.DataFrame([results])
    else:
        df = pd.read_csv(results_path)
        df = pd.concat([df, pd.DataFrame([results])], ignore_index=True)

    df.to_csv(results_path, index=False)

    # Debugging: Print final saved data
    print("test_results", pd.read_csv(results_path))
