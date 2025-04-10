from fastai.vision.all import * 
import sys # required to fix bug in os.makedirs
import os

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
    def __init__(self, img_size, backbone_name = "resnet101", datalayout = "exthf_normal"):
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
        self.weights_dir = os.path.expanduser("~/.weights") 
        self.weights_filename = "exthf_normal_resnet101_l1_unnormalized_200.pt"
        self.weights_urlbase  = "https://github.com/c3di/neural_fib_se_bse/releases/download/release_v1.0.0"
        self.load_weights()

    def load_weights( self ):
        if not os.path.exists( self.weights_dir ):
            print( "creating", self.weights_dir )
            os.makedirs( self.weights_dir )
        
        state_dict = torch.hub.load_state_dict_from_url( 
            url       = self.weights_urlbase + "/" +  self.weights_filename,
            model_dir = self.weights_dir,
            file_name = self.weights_filename )
        state_dict = torch.load( self.weights_dir + "/" + self.weights_filename )
        self.load_state_dict( state_dict )        
        
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
