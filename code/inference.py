import json
import argparse

import torch

import models.mm_classifier
import utils.dataset
import utils.eval_utils

def inference(args):

    # Debug
    if args.debug:
        args.batch_size = 3
        args.batch_size_dev = 3
        args.checkpoint = '../results/checkpoints/albef_mm.bin'
        args.name = 'debug'
        args.model = 'albef'

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Load a pretrained BLIP model and pre-processors
    model = models.mm_classifier.MMCls(args, device)
    vis_processor, txt_processor = model.get_processors()

    # Build the datasets
    img_dir = 'labeled'
    txt_path_dev_test = 'labeled/CT23_1A_checkworthy_multimodal_english_dev_test.jsonl'
    metadata_path_dev_test = 'retrieved/dev_test.json'
    dev_test_loader = utils.dataset.make_loader(txt_path_dev_test, img_dir, txt_processor, vis_processor, args.batch_size, metadata_path_dev_test, args)

    # Load from checkpoints
    print(f'loading checkpoint: {args.checkpoint}')
    loaded = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(loaded['model_state_dict'])

    # Infer / Eval
    # The official scorer does not yet accomondate task 1A...
    model.eval()
    with torch.no_grad():

        if not args.no_eval:
            n_pred, n_ref, n_hit = 0, 0, 0
        
        inference_out = []

        for idx, batch in enumerate(dev_test_loader):
            
            model_inputs = batch['model_inputs']
            out = model({
                'image': model_inputs['image'].to(device),
                'text_input': model_inputs['text_input']
            })
            
            preds = out > 0

            # Eval
            if not args.no_eval:
                labels = torch.tensor((batch['labels'])).to(device).reshape(-1, 1)

                n_pred += torch.sum(out > 0).detach().item()
                n_ref += torch.sum(labels > 0.5).detach().item()

                preds = out > 0
                hits = preds * labels
                hits = hits.detach()
                n_hit += torch.sum(hits > 0.5).detach().item()
            
            # Store predictions:
            for line_idx, text in enumerate(model_inputs['text_input']):
                line_out = {
                    'text': text,
                    'img_path': batch['image_paths'][line_idx],
                    'logit': out[line_idx][0].detach().item(),
                    'pred': preds[line_idx][0].detach().item(),
                }
                if not args.no_eval:
                    line_out['label'] = labels[line_idx][0].detach().item()
                inference_out.append(line_out)
            
            if args.debug and idx > 3:
                break
            
        # Print eval results
        if not args.no_eval:
            prec, recall, f1 = utils.eval_utils.prec_recall_f1(n_pred, n_ref, n_hit)

            print(f'#pred: {n_pred}, #true:{n_ref}, #hits:{n_hit}')
            print(f'F1: {f1}, prec: {prec}, recall: {recall}')
        
        # Store inference results
        path = f'../results/outputs/{args.name}.json'
        with open(path, 'w') as f:
            json.dump(inference_out, f)
        
        return

    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--name', default='unnamed')

    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--checkpoint', default='', type=str) 

    # Inference
    parser.add_argument('--no_eval', action='store_true')

    # Ablations: Modality
    parser.add_argument('--text_only', action='store_true')
    parser.add_argument('--image_only', action='store_true')
    parser.add_argument('--no_ocr', action='store_true')

    # Metadata
    parser.add_argument('--metadata', action='store_true')

    # Debug
    parser.add_argument('--debug', action='store_true')
    
    args = parser.parse_args()

    inference(args)