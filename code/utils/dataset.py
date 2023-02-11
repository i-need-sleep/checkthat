from PIL import Image
import jsonlines
import json
import copy

import torch
from torch.utils.data import Dataset, DataLoader

import lavis

DATA_DIR = '../data'

class MMClaimsDataset(Dataset):
    def __init__(self, txt_path, img_dir, metadata_path, args):
        
        self.txt_path = f'{DATA_DIR}/{txt_path}'
        self.img_dir = f'{DATA_DIR}/{img_dir}'
        self.args = args

        # Read the jsonl file. Store the data in a long list.
        self.data = self._prep_data()

        # Read the retrieved metadata
        if args.metadata:
            metadata_path = f'{DATA_DIR}/{metadata_path}'
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)

            metadata_stats_path = f'{DATA_DIR}/retrieved/statistics.json'
            with open(metadata_stats_path, 'r') as f:
                self.metadata_stats = json.load(f)

                if args.metadata_bin == 'mean':
                    self.metadata_stats = self.metadata_stats['mean']
                elif args.metadata_bin == 'median':
                    self.metadata_stats = self.metadata_stats['median']

    def _prep_data(self):

        data = []

        with jsonlines.open(self.txt_path) as f:
            for obj in f:
                data.append(obj)

        return data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        line = copy.deepcopy(self.data[index])

        # Concatenate the tweet and ocr texts
        text = f'{line["tweet_text"]} {line["ocr_text"] if not self.args.no_ocr else ""}'.replace('\\n', ' ')

        # Concatenate flattened metadata
        if self.args.metadata:
            line_metadata = self.metadata[line['tweet_id']]
            if 'n_likes' in line_metadata.keys() and 'n_retweets' in line_metadata.keys():
                if self.args.metadata_bin in ['mean', 'median']:
                    if line_metadata["n_likes"] > self.metadata_stats['n_likes']:
                        line_metadata["n_likes"] = 'a high number of'
                    else:
                        line_metadata["n_likes"] = 'a low number of'
                    if line_metadata["n_retweets"] > self.metadata_stats['n_retweets']:
                        line_metadata["n_retweets"] = 'a high number of'
                    else:
                        line_metadata["n_retweets"] = 'a low number of'
                text += f' This Tweet is liked {line_metadata["n_likes"]} times and retweeted {line_metadata["n_retweets"]} times.'
            if 'author_name' in line_metadata.keys():
                text += f' The author of this Tweet is {line_metadata["author_name"]}.'
            if 'verified' in line_metadata.keys():
                if line_metadata["verified"]:
                    text += ' The author is verified.'
                else:
                    text += ' The author is not verified.'
            if 'n_followers' in line_metadata.keys() and line_metadata["n_followers"] != None:
                if self.args.metadata_bin in ['mean', 'median']:
                    if line_metadata["n_followers"] > self.metadata_stats['n_followers']:
                        line_metadata["n_followers"] = 'a high number of'
                    else:
                        line_metadata["n_followers"] = 'a low number of'
                text += f' The author has {line_metadata["n_followers"]} followers.'
            if 'n_listed' in line_metadata.keys() and line_metadata["n_listed"] != None:
                if self.args.metadata_bin in ['mean', 'median']:
                    if line_metadata["n_listed"] > self.metadata_stats['n_followers']:
                        line_metadata["n_listed"] = 'a high number of'
                    else:
                        line_metadata["n_listed"] = 'a low number of'
                text += f' The author has {line_metadata["n_listed"]} Tweets.'
            if 'bio' in line_metadata.keys():
                text += f' The bio of the author is {line_metadata["bio"]}'

        # Load the raw image
        img_path = f'{self.img_dir}/{line["image_path"]}'
        img = Image.open(img_path).convert("RGB")

        # Class label
        if line['class_label'] == 'Yes':
            label = 1
        elif line['class_label'] == 'No':
            label = 0
        else:
            raise ValueError

        return {
            'text': text,
            'img': img,
            'label': label,
            'tweet_id': line['tweet_id'],
            'image_path': line["image_path"]
        }

class Collate(object):
    def __init__(self, txt_processor, vis_processor):
        
        self.txt_processor = txt_processor
        self.vis_processor = vis_processor

    def __call__(self, batch):
        texts, labels, tweet_ids, image_paths = [], [], [], []

        for line_idx, line in enumerate(batch):
            labels.append(line['label'])
            tweet_ids.append(line['tweet_id'])
            image_paths.append(line['image_path'])
            

            text = self.txt_processor(line['text'])
            texts.append(text)

            img = self.vis_processor(line['img']).unsqueeze(0)
            if line_idx == 0:
                imgs = img
            else:
                imgs = torch.concat((imgs, img), dim=0)

        return {
            'model_inputs': {
                'image': imgs,
                'text_input': texts,
            },
            'labels': labels,
            'tweeet_ids': tweet_ids,
            'image_paths': image_paths
        }

def make_loader(txt_path, img_dir, txt_processor, vis_processor, batch_size, metadata_path, args, shuffle=True):
    dataset = MMClaimsDataset(txt_path, img_dir, metadata_path, args)
    collate = Collate(txt_processor, vis_processor)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate, shuffle=shuffle)
    return loader

if __name__ == '__main__':
    txt_path = 'labeled/CT23_1A_checkworthy_multimodal_english_train.jsonl'
    img_dir = 'labeled'

    model, vis_processors, txt_processors = lavis.models.load_model_and_preprocess(name="blip_feature_extractor", model_type="base", is_eval=False, device='cpu')
    txt_processor = txt_processors['eval']
    vis_processor = vis_processors['eval']

    loader = make_loader(txt_path, img_dir, txt_processor, vis_processor, 3)
    for batch in loader:

        out = model.extract_features(batch['model_inputs'])['multimodal_embeds']
        break

    for name, param in model.named_parameters():
        print(name, param.requires_grad)