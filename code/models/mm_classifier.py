import torch

import lavis

class MMCls(torch.nn.Module):
    def __init__(self, args, device):
        super(MMCls, self).__init__()

        self.args = args

        if args.model == 'blip':
            name = 'blip_feature_extractor'
        elif args.model == 'albef':
            name = 'albef_feature_extractor'
        else:
            raise NotImplementedError

        self.model, self.vis_processors, self.txt_processors = lavis.models.load_model_and_preprocess(name=name, model_type='base', is_eval=False, device=device)

        self.cls_head = torch.nn.Linear(768, 1)
        
        self.to(device)

    def forward(self, inputs):
        
        if self.args.text_only:
            inputs = {'text_input': inputs['text_input']}
            model_out = self.model.extract_features(inputs, mode='text')
            embs = model_out['text_embeds']
        elif self.args.image_only:
            inputs = {'image': inputs['image']}
            model_out = self.model.extract_features(inputs, mode='image')
            embs = model_out['image_embeds']
        else:
            model_out = self.model.extract_features(inputs)
            embs = model_out['multimodal_embeds']

        if self.args.pooling == '':
            # Slice off the first emb dim
            cls_in = embs[:, 0, :]
        elif self.args.pooling == 'mean':
            cls_in = torch.mean(embs, dim=1)
        else:
            raise NotImplementedError

        out = self.cls_head(cls_in)
        
        return out
    
    def get_processors(self):
        return self.vis_processors['eval'], self.txt_processors['eval']