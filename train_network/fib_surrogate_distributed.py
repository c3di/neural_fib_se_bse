import sys
import pathlib
import matplotlib as mp
import argparse

import torch
import fastai
from fastai.vision.all import * 
from fastai import metrics
import tifffile
import numpy

print("sys.version", sys.version)
print("cuda device name(0)", torch.cuda.get_device_name(0))
print("torch.__version__", torch.__version__)
print("fastai.__version__", fastai.__version__)

def get_items(input_path):
    file_names = get_image_files(input_path)
    file_names = [filename for filename in file_names if "_gt.tif" in str( filename ) ]
    return file_names

def get_hf_0(filename):
    filename_parts = str(filename.stem).split("_")
    y = filename.parent / Path( "_".join(filename_parts[:-1]) + "_entry_hf_0.tif" )
    return str(y)

def get_hf_1(filename):
    filename_parts = str(filename.stem).split("_")
    y = filename.parent / Path( "_".join(filename_parts[:-1]) + "_exit_hf_0.tif" )
    return str(y)

def get_hf_2(filename):
    filename_parts = str(filename.stem).split("_")
    y = filename.parent / Path( "_".join(filename_parts[:-1]) + "_entry_hf_1.tif" )
    return str(y)

def get_hf_3(filename):
    filename_parts = str(filename.stem).split("_")
    y = filename.parent / Path( "_".join(filename_parts[:-1]) + "_exit_hf_1.tif" )
    return str(y)

def get_normal(filename):
    filename_parts = str(filename.stem).split("_")
    y = filename.parent / Path( "_".join(filename_parts[:-1]) + "_normal.tif" )
    return str(y)

def get_bse(filename):
    filename_parts = str(filename.stem).split("_")
    y = filename.parent / Path( "_".join(filename_parts[:-1]) + "_bse.tif" )
    return str(y)

def get_se(filename):
    filename_parts = str(filename.stem).split("_")
    y = filename.parent / Path( "_".join(filename_parts[:-1]) + "_se.tif" )
    return str(y)

@typedispatch
def show_batch(x:tuple, y, samples, ctxs=None, max_n=6, nrows=None, ncols=2, figsize=None, **kwargs):    
    hf0,hf1,hf2,hf3,normal = x
    
    show_output = False
    bse,se = y
    batch_size = hf0.shape[0]
    
    nrows = min(x[0].shape[0], max_n)
    if show_output:
        ncols = 7
    else:
        ncols = 5
    if ctxs is None: ctxs = get_grid(nrows*ncols, nrows=nrows, ncols=ncols, figsize=figsize)
    
    ctxi = 0
    for i in range(batch_size):
        hf0_image = hf0[i,:,:,:].squeeze(0)
        show_image(hf0_image, title="heightmap (entry 1)", ctx=ctxs[ctxi], cmap="gray", **kwargs)
        
        ctxi = ctxi+1
        
        hf1_image = hf1[i,:,:,:].squeeze(0)
        show_image(hf1_image, title="heightmap (exit 1)", ctx=ctxs[ctxi], cmap="gray", **kwargs)
        ctxi = ctxi+1

        hf2_image = hf2[i,:,:,:].squeeze(0)
        show_image(hf2_image, title="heightmap (entry 2) ", ctx=ctxs[ctxi], cmap="gray", **kwargs)
        ctxi = ctxi+1
        
        hf3_image = hf3[i,:,:,:].squeeze(0)
        show_image(hf3_image, title="heightmap (exit 2) ", ctx=ctxs[ctxi], cmap="gray", **kwargs)
        ctxi = ctxi+1        
        
        normal_image = normal[i,:,:,:].squeeze(0)      
        show_image(normal_image, title="normalmap", ctx=ctxs[ctxi], **kwargs)
        ctxi = ctxi+1
        
        if show_output:
            bse_image = bse[i,:,:,:].squeeze(0)      
            show_image(bse_image, title="BSE", ctx=ctxs[ctxi], **kwargs)
            ctxi = ctxi+1

            se_image = se[i,:,:,:].squeeze(0)      
            show_image(se_image, title="SE", ctx=ctxs[ctxi], **kwargs)
            ctxi = ctxi+1
    plt.savefig("data_representation.png", dpi=300)  
    
def create_inner_model( dataloader, backbone, datalayout, **kwargs ):
    img_size = dataloader.one_batch()[0].shape[-2:]
    
    if datalayout == "exthf_normal" or datalayout == "normal_exthf":
        n_in=7
    if datalayout == "exthf_only" or datalayout == "hf_normal":
        n_in=4
    if datalayout == "hf_only":
        n_in=1
    
    model = create_unet_model(backbone, 2, img_size, n_in=n_in, **kwargs)
    return model

class FIBModel(torch.nn.Module):
    def __init__(self, inner_model, datalayout = "normal_exthf"):
        super().__init__()
        self.inner_model = inner_model
        self.datalayout = datalayout
        
    def forward(self, x_hf_0, x_hf_1, x_hf_2, x_hf_3, x_normal):
        if self.datalayout == "normal_exthf":
            x = torch.cat( (x_normal, x_hf_0, x_hf_1, x_hf_2, x_hf_3 ), dim=1 )
        elif self.datalayout == "exthf_normal":
            x = torch.cat( (x_hf_0, x_hf_1, x_hf_2, x_hf_3, x_normal ), dim=1 )
        elif self.datalayout == "exthf_only":
            x = torch.cat( (x_hf_0, x_hf_1, x_hf_2, x_hf_3 ), dim=1 )
        elif self.datalayout == "hf_normal":
            x = torch.cat( (x_hf_0, x_normal ), dim=1 )
        elif self.datalayout == "hf_only":
            x = x_hf_0
        output_of_inner_model = self.inner_model(x)
        output = torch.split(output_of_inner_model, 1, dim=1)
        # output = output_of_inner_model
        return output

class CombinedLoss():
    def __init__(self, losses, weights=None, reduction='mean', axis=-1):
        self.losses = losses
        self._reduction = reduction
        self.axis = axis
        self.weights = weights
        if weights is None:
            self.weights = []
            for _ in losses:
                self.weights.append(1.0)
        
    def __call__(self, out, *yb):
        total_loss = 0.0
        # out = torch.split(out, 1, dim=1)
        for i,loss_fct in enumerate( self.losses ):
            loss_fct.reduction = 'none'
            total_loss += loss_fct(out[i], yb[i]) * self.weights[i]
        if self.reduction == "mean":
            total_loss = total_loss.mean()
        elif self.reduction == "sum":
            total_loss = total_loss.sum()
        return total_loss
    
    @property
    def reduction(self) -> str:
        return self._reduction    
    
    @reduction.setter
    def reduction(self, reduction:str):
        self._reduction = reduction  
    
    def decodes(self, x:Tensor) -> Tensor:    
        return x
        # return x.argmax(dim=self.axis)
    
    def activation(self, x:Tensor) -> Tensor:                 
        activation = torch.zeros(x[0].shape)
        for xi in list(x):
            activation += F.softmax(xi, dim=self.axis)    
        return activation     
    
def total_mse(inp, *targ):
    total = 0.0
    for i in range(len(inp)):
        total += mse(inp[i], targ[i])
    return total

def total_l1(inp, *targ):
    total = 0.0
    for i in range(len(inp)):
        total += mae(inp[i], targ[i])
    return total       
    
def filler(depth = 0):
    result = ""
    for i in range(depth):
        result = result + "  "
    return result

def print_tuple(x, depth = 0):
    if isinstance(x,fastai.torch_core.TensorBase):
        print(filler(depth), type(x), x.shape)
    elif isinstance(x,torch.Tensor):
        print(filler(depth), type(x), x.shape)
    elif type(x) is tuple:
        print(filler(depth), type(x), len(x) )
        for child_x in list(x):
            print_tuple( child_x, depth+1 )
    else:
        print(filler(depth), type(x))

class ConstantFunc():
    "Returns a function that returns `o`"
    def __init__(self, o): self.o = o
    def __call__(self, *args, **kwargs): return self.o        
        
class PredictionsFromTupleCallback(Callback):    
    def before_validate(self):
        self.preds = []
        self.targets = []
            
    def after_pred(self, **kwargs:Any)->None:
        se,bse = self.pred
        se = to_detach(se)
        bse = to_detach(bse)
        self.preds.append((se,bse))
        self.targets.append(self.yb)       
        
def create_top_loss_image( learner, output_filename ):    
    interpretation = Interpretation.from_learner( learner )    
    values,indices = interpretation.top_losses(k=4)

    metrics_string = ""
    for metric in learner.metrics:
        metrics_string = metrics_string + metric.name + ": " + "{:.5f}".format(metric.value.item() ) + "\n"
    
    tmp_data_loader = learner.dls[1].new( get_idxs = ConstantFunc( indices ), bs=1 )
    cb = PredictionsFromTupleCallback()
    ctx_mgrs = learner.validation_context(cbs=[cb])
    with ContextManagers(ctx_mgrs):
        learner._do_epoch_validate(dl=tmp_data_loader)

    all_predictions = cb.preds
    all_targets     = cb.targets

    figure      = plt.figure( constrained_layout=True, figsize=(16,48) )
    figure.suptitle(metrics_string, fontsize=16 )
    
    sub_figures = figure.subfigures(nrows=4, ncols=1)
    
    # fig, axs = plt.subplots(2*4, 3, figsize=(16,48))

    for i,idx in enumerate(indices):
        filename = str( learner.dls.valid_ds.items[idx] )
        
        sub_figures[i].suptitle( filename )
        axs = sub_figures[i].subplots(nrows=2, ncols=3)        
        
        # hf_preds,bse_preds = preds
        se_pred,bse_pred = all_predictions[i]
        se_pred  = torch.squeeze( se_pred, 0 )
        bse_pred = torch.squeeze( bse_pred, 0 )

        se_target,bse_target = all_targets[i]
        se_target  = torch.squeeze(  se_target, 0 )
        bse_target = torch.squeeze( bse_target, 0 )

        se_resid  = abs(se_pred.cpu() - se_target.cpu())
        bse_resid = abs(bse_pred.cpu() - bse_target.cpu())

        show_image( ax=axs[0,0], im=se_pred,        title="SE Prediction ", vmin=0, vmax=1,  cmap="gray")
        show_image( ax=axs[1,0], im=bse_pred,       title="BSE Prediction", vmin=0, vmax=1,  cmap="gray")

        show_image( ax=axs[0,1], im=se_target,      title="SE Target", vmin=0, vmax=1,       cmap="gray")
        show_image( ax=axs[1,1], im=bse_target,     title="BSE Target", vmin=0, vmax=1,      cmap="gray")

        show_image( ax=axs[0,2], im=se_resid,       title="SE Residual", vmin=0, vmax=0.20,  cmap="coolwarm")
        show_image( ax=axs[1,2], im=bse_resid,      title="BSE Residual", vmin=0, vmax=0.20, cmap="coolwarm")

    plt.savefig(output_filename, dpi=300)  
   

class PILImageBW16( PILImageBW ):
    @classmethod
    def create( cls, fn, mode=None):
        data = tifffile.imread(fn)
        data = data.astype( numpy.float32 )
        image = Image.fromarray( data )
        return cls( image )
 
def train_neural_network( input_path = Path('/training-data'), 
                          do_train = False, 
                          do_evaluate = True, 
                          do_test = False, 
                          n_epochs = [19,30,50],
                          backbone_name = "resnet152",
                          datalayout = "exthf_normal",
                          lossname = "l1"):
    
    if backbone_name == "resnet152":
        backbone = resnet152
    elif backbone_name == "resnet101":
        backbone = resnet101
    elif backbone_name == "resnet50":
        backbone = resnet50
    elif backbone_name == "resnet34":
        backbone = resnet34
    
    if lossname == "l1":
        loss = CombinedLoss([ L1LossFlat(), L1LossFlat()] )    
    elif lossname == "l2":
        loss = CombinedLoss([ MSELossFlat(), MSELossFlat()] )
    elif lossname == "l1_weighted":
        loss = CombinedLoss([ L1LossFlat(), L1LossFlat()], [1.0, 1.691] )
    elif lossname == "l2_weighted":
        loss = CombinedLoss([ MSELossFlat(), MSELossFlat()], [1.0, 1.691] )
        
    print("training network with settings ", datalayout, backbone_name, lossname )
    print("                               ", backbone, loss )

    file_names = get_items(input_path)      
    item_transforms  = [RandomCrop((512,512)),DihedralItem]
    datablocks = DataBlock( blocks=(ImageBlock(cls=PILImageBW16),
                                    ImageBlock(cls=PILImageBW16), 
                                    ImageBlock(cls=PILImageBW16), 
                                    ImageBlock(cls=PILImageBW16), 
                                    ImageBlock, 
                                    ImageBlock(cls=PILImageBW), 
                                    ImageBlock(cls=PILImageBW)),
                            n_inp=5,
                            get_items=get_items,
                            getters=[get_hf_0, get_hf_1, get_hf_2, get_hf_3, get_normal, get_bse, get_se],
                            splitter=RandomSplitter(valid_pct=0.2, seed=42),
                            item_tfms=item_transforms)

    data_loader = datablocks.dataloaders(input_path, bs=3, num_workers=0 )

    learning_rate=0.001
  
    model = FIBModel( create_inner_model( data_loader, backbone, datalayout ), datalayout )
    learner = Learner( data_loader, model, loss_func=loss, metrics=[total_mse, total_l1] )

    if do_train:
        print("training neural network for", n_epochs, "epochs total")
        learner.fit( 1, lr=learning_rate )
        epochs = 1

        for step in n_epochs:
            epochs = epochs + step
            weight_name = datalayout + "_" + backbone_name + "_" + lossname + "_" + str(epochs)
            print(weight_name)
            learner.fit( step, lr=learning_rate, cbs=[ShowGraphCallback(), CSVLogger(fname=weight_name+".csv")] )    
            learner.save( weight_name )
    else:
        epochs = 100
        weight_name = datalayout + "_" + backbone_name + "_" + lossname + "_" + str(epochs)
        print("skipping training of neural network, loading", weight_name )
        learner.load(weight_name)

    if do_evaluate:
        print("creating top loss image", weight_name)   
        create_top_loss_image( learner, weight_name + "_top4.png")

    if do_test:
        test_input_path = Path('./test_data')
        test_files = get_items( test_input_path )
        test_dataloader = learner.dls.test_dl( test_files )

        cb = PredictionsFromTupleCallback()
        ctx_mgrs = learner.validation_context(cbs=[cb])
        with ContextManagers(ctx_mgrs):
            learner._do_epoch_validate(dl=test_dataloader)
        all_preds = cb.preds
            

parser = argparse.ArgumentParser()
parser.add_argument(
    "-nr", "--nr", default=0, type=int, help="ranking within the nodes"
)
args = parser.parse_args()

backbones   = [ "resnet152", "resnet101", "resnet50", "resnet34" ]
datalayouts = [ "normal_exthf", "exthf_normal", "exthf_only", "hf_normal", "hf_only" ]
losses      = [ "l1", "l2", "l1_weighted", "l2_weighted" ]

iterate_layouts    =   [ (layout, "resnet152", "l1") for layout in datalayouts ]
iterate_backbones = [ ("exthf_normal", backbone, "l1") for backbone in backbones ]
iterate_losses    =   [ ("exthf_normal", "resnet152", loss) for loss in losses ]

experiments = iterate_layouts + iterate_backbones + iterate_losses
print(experiments)

data_layout,backbone_name,loss_name = experiments[args.nr]

train_neural_network(backbone_name = backbone_name,
                     datalayout = data_layout,
                     lossname = loss_name)                     
