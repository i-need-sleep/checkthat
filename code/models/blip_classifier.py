import torch

import lavis

class BLIPCls(torch.nn.Module):
    def __init__(self, device):
        super(BLIPCls, self).__init__()

        self.blip, self.vis_processors, self.txt_processors = lavis.models.load_model_and_preprocess(name="blip_feature_extractor", model_type="base", is_eval=False, device=device)

        self.cls_head = torch.nn.Linear(768, 1)
        
        self.to(device)

    def forward(self, blip_inputs):
        
        blip_out = self.blip.extract_features(blip_inputs)
        mm_embs = blip_out['multimodal_embeds']

        # Slice off the first emb dim
        cls_in = mm_embs[:, 0, :]

        out = self.cls_head(cls_in)
        
        return out
    
    def get_processors(self):
        return self.vis_processors['eval'], self.txt_processors['eval']