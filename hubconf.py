dependencies = ['torch']
from simulator import *

def fib_se_bse( weights=None **kwargs ):
    """ # FIB SE BSE Model
    default weights are Resnet 152 backbone and exthf-normal datalayout
    """
    model = new FIBModel(**kwargs)
    
    if pretrained:
        # For checkpoint saved in local GitHub repo, e.g. 
        weight_name = datalayout + "_" + backbone_name + "_" + lossname + "_" + str(epochs)
        path_to_checkpoint = "weights/" + weights + ".pth"
        dirname = os.path.dirname(__file__)
        checkpoint = os.path.join(dirname, path_to_checkpoint )
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict)
    
    return model

