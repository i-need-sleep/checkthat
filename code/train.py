import os
import argparse
import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

import models.blip_classifier
import utils.dataset
import utils.eval_utils

def train(args):
    
    print(args)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Load a pretrained BLIP model and pre-processors
    model = models.blip_classifier.BLIPCls(device)
    vis_processor, txt_processor = model.get_processors()

    # Build the datasets
    img_dir = 'labeled'
    txt_path_train = 'labeled/CT23_1A_checkworthy_multimodal_english_train.jsonl'
    txt_path_dev = 'labeled/CT23_1A_checkworthy_multimodal_english_dev.jsonl'
    train_loader = utils.dataset.make_loader(txt_path_train, img_dir, txt_processor, vis_processor, args.batch_size)
    dev_loader = utils.dataset.make_loader(txt_path_dev, img_dir, txt_processor, vis_processor, args.batch_size_dev)

    # Setup Tensorboard
    date_str = str(datetime.datetime.now())[:-7].replace(':','-')
    writer = SummaryWriter(log_dir=f'../results/runs/{args.name}/batch_size={args.batch_size}, Adam_lr={args.lr}/{date_str}', comment=f'{args.name}, batch_size={args.batch_size}, Adam_lr_enc={args.lr}, {date_str}')

    # CE losses. TODO: try a weighted focal loss
    criterion = torch.nn.BCEWithLogitsLoss()

    # Optimizer
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Load from checkpoints
    if args.checkpoint != '':
        print(f'loading checkpoint: {args.checkpoint}')
        loaded = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(loaded['model_state_dict'])
        optimiser.load_state_dict(loaded['optimiser_state_dict'])
        optimiser.param_groups[0]['capturable'] = True

    # train
    n_iter = 0
    n_prev_iter = 0
    running_loss = 0
    best_f1 = 0
    print('Training...')

    for epoch in range(args.n_epoch):
        print(f'Epoch: {epoch}')

        for batch_idx, batch in enumerate(train_loader):

            model.train()
            optimiser.zero_grad()

            model_inputs = batch['model_inputs']
            out = model({
                'image': model_inputs['image'].to(device),
                'text_input': model_inputs['text_input']
            })

            # Loss
            labels = torch.tensor((batch['labels'])).to(device).reshape(-1, 1).float()
            
            loss = criterion(out, labels)
            
            loss.backward()
            optimiser.step()

            n_iter += 1
            writer.add_scalar('loss/cls_loss', loss, n_iter)
            running_loss += loss.detach()

        step_loss = running_loss / (n_iter - n_prev_iter)
        print(f'Training loss: {step_loss}')
        n_prev_iter = n_iter
        running_loss = 0

        # Eval
        if epoch % 10 == 0:

            prec, recall, f1 = eval(model, dev_loader, device)
            writer.add_scalar('dev/prec', prec, n_iter)
            writer.add_scalar('dev/recall', recall, n_iter)
            writer.add_scalar('dev/f1', f1, n_iter)
            print(f'F1: {f1}')

            if f1 > best_f1:
                best_f1 = f1

            try:
                os.makedirs(f'../results/checkpoints/{args.name}')
            except:
                pass

            save_path = f'../results/checkpoints/{args.name}/batchsize{args.batch_size}_lr{args.lr}_{epoch}_{batch_idx}_{best_f1}.bin'

            print(f'Best f1: {best_f1}')
            print(f'Saving the checkpoint at {save_path}')
            torch.save({
                'epoch': epoch,
                'step': n_iter,
                'model_state_dict': model.state_dict(),
                'optimiser_state_dict': optimiser.state_dict(),
                }, save_path)
                    
    print('DONE !!!')
    return

def eval(model, loader, device):
    
    model.eval()
    with torch.no_grad():
        n_pred, n_ref, n_hit = 0, 0, 0

        for idx, batch in enumerate(loader):
            
            model_inputs = batch['model_inputs']
            out = model({
                'image': model_inputs['image'].to(device),
                'text_input': model_inputs['text_input']
            })

            labels = torch.tensor((batch['labels'])).to(device).reshape(-1, 1)

            n_pred += torch.sum(out > 0).detach().item()
            n_ref += torch.sum(labels > 0.5).detach().item()

            preds = out > 0
            hits = preds * labels
            hits = hits.detach()
            n_hit += torch.sum(hits > 0.5).detach().item()
            
        prec, recall, f1 = utils.eval_utils.prec_recall_f1(n_pred, n_ref, n_hit)

        print(f'#pred: {n_pred}, #true:{n_ref}, #hits:{n_hit}')
        print(f'F1: {f1}, prec: {prec}, recall: {recall}')
        
        return prec, recall, f1

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--name', default='unnamed')

    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--batch_size_dev', default=16, type=int)

    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--n_epoch', default=1000, type=int)
    parser.add_argument('--checkpoint', default='', type=str) 

    args = parser.parse_args()

    train(args)
