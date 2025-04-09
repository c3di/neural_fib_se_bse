from fastai.vision.all import * 
import PILImageBW16

def get_image_size_from_dataloader( dataloader ):
    return dataloader.one_batch()[0].shape[-2:]

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

class PILImageBW16( PILImageBW ):
    @classmethod
    def create( cls, fn, mode=None):
        data = tifffile.imread(fn)
        data = data.astype( numpy.float32 )
        image = Image.fromarray( data )
        return cls( image )

def create_dataloader( input_path = Path('/training-data'), 
                       datalayout = "exthf_normal",
                       bs=3):
    
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

    data_loader = datablocks.dataloaders(input_path, bs=bs, num_workers=0 )
    return data_loader

def create_inner_model( dataloader, backbone, img_size, datalayout, **kwargs ):
    if datalayout == "exthf_normal" or datalayout == "normal_exthf":
        n_in=7
    if datalayout == "exthf_only" or datalayout == "hf_normal":
        n_in=4
    if datalayout == "hf_only":
        n_in=1
    
    model = create_unet_model(backbone, 2, img_size, n_in=n_in, **kwargs)
    return model

class FIBModel(torch.nn.Module):
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
        self.inner_model = create_inner_model( backbone, img_size, datalayout )
        
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
