from PIL import Image
import jsonlines

import torch
from torch.utils.data import Dataset, DataLoader

import lavis

DATA_DIR = '../data'

class MMClaimsDataset(Dataset):
    def __init__(self, txt_path, img_dir):
        
        self.txt_path = f'{DATA_DIR}/{txt_path}'
        self.img_dir = f'{DATA_DIR}/{img_dir}'


        # Read the jsonl file. Store the data in a long list.
        self.data = self._prep_data()

    def _prep_data(self):

        data = []

        with jsonlines.open(self.txt_path) as f:
            for obj in f:
                data.append(obj)

        return data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        line = self.data[index]

        # Concatenate the tweet and ocr texts
        text = f'{line["tweet_text"]} {line["ocr_text"]}'.replace('\\n', ' ')

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
            'tweet_id': line['tweet_id']
        }

class Collate(object):
    def __init__(self, txt_processor, vis_processor):
        
        self.txt_processor = txt_processor
        self.vis_processor = vis_processor

    def __call__(self, batch):
        texts, labels, tweet_ids = [], [], []

        for line_idx, line in enumerate(batch):
            labels.append(line['label'])
            tweet_ids.append(line['tweet_id'])

            text = self.txt_processor(line['text'])
            texts.append(text)

            img = self.vis_processor(line['img']).unsqueeze(0)
            if line_idx == 0:
                imgs = img
            else:
                imgs = torch.concat((imgs, img), dim=0)

        return {
            'model_inputs': {
                'image': imgs, # Need debugging. This should be a tensor.
                'text_input': texts,
            },
            'labels': labels,
            'tweeet_ids': tweet_ids
        }

def make_loader(txt_path, img_dir, txt_processor, vis_processor, batch_size, shuffle=True):
    dataset = MMClaimsDataset(txt_path, img_dir)
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