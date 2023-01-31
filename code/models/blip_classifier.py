import torch

import lavis

class BLIPCls(torch.nn.Module):
    def __init__(self, args, device):
        super(BLIPCls, self).__init__()

        self.args = args

        self.blip, self.vis_processors, self.txt_processors = lavis.models.load_model_and_preprocess(name="blip_feature_extractor", model_type="base", is_eval=False, device=device)

        self.cls_head = torch.nn.Linear(768, 1)
        
        self.to(device)

    def forward(self, blip_inputs):
        
        if self.args.text_only:
            blip_inputs = {'text_input': blip_inputs['text_input']}
            blip_out = self.blip.extract_features(blip_inputs, mode='text')
            embs = blip_out['text_embeds']
        elif self.args.image_only:
            blip_inputs = {'image': blip_inputs['image']}
            blip_out = self.blip.extract_features(blip_inputs, mode='image')
            embs = blip_out['image_embeds']
        else:
            blip_out = self.blip.extract_features(blip_inputs)
            embs = blip_out['multimodal_embeds']

        # Slice off the first emb dim
        cls_in = embs[:, 0, :]

        out = self.cls_head(cls_in)
        
        return out
    
    def get_processors(self):
        return self.vis_processors['eval'], self.txt_processors['eval']