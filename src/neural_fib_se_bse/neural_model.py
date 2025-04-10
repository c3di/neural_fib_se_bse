from fastai.vision.all import * 

def create_inner_model( backbone, img_size, datalayout, **kwargs ):
    if datalayout == "exthf_normal" or datalayout == "normal_exthf":
        n_in=7
    if datalayout == "exthf_only" or datalayout == "hf_normal":
        n_in=4
    if datalayout == "hf_only":
        n_in=1
    
    model = create_unet_model(backbone, 2, img_size, n_in=n_in, **kwargs)
    return model

class NeuralModel(torch.nn.Module):
    def __init__(self, img_size, backbone_name = "resnet101", datalayout = "normal_exthf"):
        super().__init__()
        self.datalayout = datalayout
        self.backbone = None
        if backbone_name == "resnet152":
            self.backbone = resnet152
        elif backbone_name == "resnet101":
            self.backbone = resnet101
        elif backbone_name == "resnet50":
            self.backbone = resnet50
        elif backbone_name == "resnet34":
            self.backbone = resnet34
        self.inner_model = create_inner_model( self.backbone, img_size, self.datalayout )
        
    def forward(self, x_hf, x_normal):
        if self.datalayout == "normal_exthf":
            x = torch.cat( (x_normal, x_hf ), dim=1 )
        elif self.datalayout == "exthf_normal":
            x = torch.cat( (x_hf, x_normal ), dim=1 )
        elif self.datalayout == "exthf_only":
            x = x_hf
        elif self.datalayout == "hf_normal":
            x = x_normal
        elif self.datalayout == "hf_only":
            x = x_hf[0]
            x = x.unsqueeze( 0 )
        output_of_inner_model = self.inner_model(x)
        output = torch.split(output_of_inner_model, 1, dim=1)
        return output
