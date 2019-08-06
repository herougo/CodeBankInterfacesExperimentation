# pip install torchvision

import torch
import torchvision
import inspect

def recursive_filter(obj, filter_fn, children_fn, use_children_fn=None, max_depth=5):
    # includes internal nodes
    if filter_fn(obj):
        yield obj
    use_children = not use_children_fn or use_children_fn(obj)
    if max_depth > 0 and use_children:
        for child in children_fn(obj):
            for result in recursive_filter(child, filter_fn, children_fn, use_children_fn, max_depth - 1):
                yield result

def get_attrs(x):
    try:
        return [getattr(x, attr) for attr in dir(x) if hasattr(x, attr) and not attr.startswith('_')]
    except ImportError:  # except Exception as ex is not enough to catch this for some reason
        return []

def is_dataset_subclass(x):
    return inspect.isclass(x) and issubclass(x, torch.utils.data.Dataset)

def is_python_module(x):
    import types
    return isinstance(x, types.ModuleType)

def get_class_absolute_path(obj_or_class):
    import inspect
    """ ie get_class_absolute_path(Caltech101) -> torchvision.datasets.caltech.Caltech101 """
    # https://stackoverflow.com/questions/2020014/get-fully-qualified-class-name-of-an-object-in-python
    # o.__module__ + "." + o.__class__.__qualname__ is an example in
    # this context of H.L. Mencken's "neat, plausible, and wrong."
    # Python makes no guarantees as to whether the __module__ special
    # attribute is defined, so we take a more circumspect approach.
    # Alas, the module name is explicitly excluded from __qualname__
    # in Python 3.
    
    if not inspect.isclass(obj_or_class):
        c = obj_or_class.__class__
    else:
        c = obj_or_class

    module = c.__module__
    if module is None or module == str.__class__.__module__:
        return c.__name__  # Avoid reporting __builtin__
    else:
        return module + '.' + c.__name__
    
def pp(thing, indent=4):
    import pprint
    pprint.PrettyPrinter(indent=indent).pprint(thing)
    
            
all_datasets = list(recursive_filter(
    torchvision.datasets, 
    filter_fn=is_dataset_subclass,
    children_fn=get_attrs,
    use_children_fn=is_python_module))

print(all_datasets)

pp(set([get_class_absolute_path(d) for d in all_datasets]))


KNOWN_USED_DATASET_CLASSES = {
    'torchvision.datasets.caltech.Caltech101',
    'torchvision.datasets.caltech.Caltech256',
    'torchvision.datasets.celeba.CelebA',
    'torchvision.datasets.cifar.CIFAR10',
    'torchvision.datasets.cifar.CIFAR100',
    'torchvision.datasets.cityscapes.Cityscapes',
    'torchvision.datasets.coco.CocoCaptions',
    'torchvision.datasets.coco.CocoDetection',
    'torchvision.datasets.flickr.Flickr30k',
    'torchvision.datasets.flickr.Flickr8k',
    'torchvision.datasets.imagenet.ImageNet',
    'torchvision.datasets.lsun.LSUN',
    'torchvision.datasets.lsun.LSUNClass',
    'torchvision.datasets.mnist.EMNIST',
    'torchvision.datasets.mnist.FashionMNIST',
    'torchvision.datasets.mnist.KMNIST',
    'torchvision.datasets.mnist.MNIST',
    'torchvision.datasets.omniglot.Omniglot',
    'torchvision.datasets.phototour.PhotoTour',
    'torchvision.datasets.sbd.SBDataset',
    'torchvision.datasets.sbu.SBU',
    'torchvision.datasets.semeion.SEMEION',
    'torchvision.datasets.stl10.STL10',
    'torchvision.datasets.svhn.SVHN',
    'torchvision.datasets.voc.VOCDetection',
    'torchvision.datasets.voc.VOCSegmentation'}

KNOWN_UNUSED_DATASET_CLASSES = [
    'torch.utils.data.dataset.ConcatDataset',
    'torch.utils.data.dataset.Dataset',
    'torch.utils.data.dataset.Subset',
    'torch.utils.data.dataset.TensorDataset',
    'torchvision.datasets.vision.VisionDataset',
    'torchvision.datasets.folder.DatasetFolder',
    'torchvision.datasets.folder.ImageFolder',
    'torchvision.datasets.fakedata.FakeData'
]

import os

TRAIN = 0
VAL = 1
TEST = 2
ALL = -1

TORCH_SPLIT_MAPPING = {
    TRAIN: 'train',
    VAL: 'val',
    TEST: 'test',
    ALL: 'all'
}


class TorchVisionDatasetWrapper:
    def __init__(self, dataset_class, data_dir, split='all', download=False, **kwargs):
        if get_class_absolute_path(dataset_class) not in KNOWN_USED_DATASET_CLASSES:
            raise ValueError('Unsupported class: {}'.format(datset_class))
            
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        self._kwargs = {'dataset_class': dataset_class, 'data_dir': data_dir,
                        'split': split, 'download': download, **kwargs}
        split = 'all'
        # TODO: refactor
        try:
            self._data = dataset_class(root=data_dir, split='all', download=download, **kwargs)
        except:
            self._data = dataset_class(root=data_dir, download=download, **kwargs)
            
    def get_shape(self):
        raise NotImplementedError()
            
    def get_nth_example(self, index):
        return self._data[index]
            
    def get_n_examples(self):
        return len(self._data)  # CORRECT??
    
    def download_and_prepare(self):
        self._data.download()
        
    def as_pytorch_dataset(self, split=TRAIN):
        kwargs = dict(self._kwargs)
        kwargs['split'] = TORCH_SPLIT_MAPPING[split]
        kwargs['download'] = False
        return TorchVisionDatasetWrapper(**self._kwargs)._data

    def as_pytorch_dataloader(self, split=TRAIN, **kwargs):
        dataset = self.as_pytorch_dataset(split=TRAIN)
        return torch.utils.data.DataLoader(dataset, **kwargs)


def is_pytorch_module_subclass(x):
    return inspect.isclass(x) and issubclass(x, torch.nn.Module)

all_models = list(recursive_filter(
    torchvision.models, 
    filter_fn=lambda x: is_pytorch_module_subclass(x) and get_class_absolute_path(x).startswith('torchvision.models'),
    children_fn=get_attrs,
    use_children_fn=lambda x: is_python_module(x)))

all_model_paths = set([get_class_absolute_path(m) for m in all_models])

print(all_models)
pp(all_model_paths)

ALL_KNOWN_UNSORTED_MODULE_PATHS = {
    'torchvision.models.densenet.DenseNet',
    'torchvision.models.detection.backbone_utils.BackboneWithFPN',
    'torchvision.models.detection.faster_rcnn.FastRCNNPredictor',
    'torchvision.models.detection.faster_rcnn.FasterRCNN',
    'torchvision.models.detection.faster_rcnn.TwoMLPHead',
    'torchvision.models.detection.generalized_rcnn.GeneralizedRCNN',
    'torchvision.models.detection.keypoint_rcnn.KeypointRCNN',
    'torchvision.models.detection.keypoint_rcnn.KeypointRCNNHeads',
    'torchvision.models.detection.keypoint_rcnn.KeypointRCNNPredictor',
    'torchvision.models.detection.mask_rcnn.MaskRCNN',
    'torchvision.models.detection.mask_rcnn.MaskRCNNHeads',
    'torchvision.models.detection.mask_rcnn.MaskRCNNPredictor',
    'torchvision.models.detection.roi_heads.RoIHeads',
    'torchvision.models.detection.rpn.AnchorGenerator',
    'torchvision.models.detection.rpn.RPNHead',
    'torchvision.models.detection.rpn.RegionProposalNetwork',
    'torchvision.models.detection.transform.GeneralizedRCNNTransform',
    'torchvision.models.googlenet.GoogLeNet',
    'torchvision.models.inception.BasicConv2d',
    'torchvision.models.inception.Inception3',
    'torchvision.models.inception.InceptionA',
    'torchvision.models.inception.InceptionAux',
    'torchvision.models.inception.InceptionB',
    'torchvision.models.inception.InceptionC',
    'torchvision.models.inception.InceptionD',
    'torchvision.models.inception.InceptionE',
    'torchvision.models.mobilenet.ConvBNReLU',
    'torchvision.models.mobilenet.InvertedResidual',
    'torchvision.models.mobilenet.MobileNetV2',
    'torchvision.models.resnet.BasicBlock',
    'torchvision.models.resnet.Bottleneck',
    'torchvision.models.resnet.ResNet',
    'torchvision.models.segmentation.deeplabv3.ASPP',
    'torchvision.models.segmentation.deeplabv3.ASPPConv',
    'torchvision.models.segmentation.deeplabv3.ASPPPooling',
    'torchvision.models.segmentation.deeplabv3.DeepLabHead',
    'torchvision.models.segmentation.deeplabv3.DeepLabV3',
    'torchvision.models.segmentation.fcn.FCN',
    'torchvision.models.segmentation.fcn.FCNHead',
    'torchvision.models.shufflenetv2.InvertedResidual',
    'torchvision.models.shufflenetv2.ShuffleNetV2',
    'torchvision.models.squeezenet.Fire',
    'torchvision.models.squeezenet.SqueezeNet',
    'torchvision.models.vgg.VGG'}

ALL_KNOWN_UNUSED_MODEL_PATHS = {
    'torchvision.models._utils.IntermediateLayerGetter'
}

ALL_KNOWN_USED_MODEL_PATHS = {
    'torchvision.models.alexnet.AlexNet',
    'torchvision.models.squeezenet.SqueezeNet',
    'torchvision.models.vgg.VGG'
}
ALL_KNOWN_USED_MODEL_PRETRAINED_PATHS = {
    'torchvision.models.alexnet.AlexNet': None,
    'torchvision.models.squeezenet.SqueezeNet': None,
    'torchvision.models.vgg.VGG': None
}

CONFIG_NUM_EPOCHS = 'num_epochs'
OPTIMIZER_MAP = {
    'adam': torch.optim.Adam
}

def parse_config(config, keys):
    config_keys = set(config.keys())
    missing = set(keys) - config_keys
    if len(missing) > 0:
        raise ValueError('Missing config varibles {} in {}'.format(missing, config))
    return [config[key] for key in keys]

def get_applicable_init_kwargs(a_class, config):
    # TODO: check what config parameters can be passed to the optimizer class
    return {}

def get_optimizer(opt, parameters, config):
    if isinstance(opt, torch.nn.Module):
        return opt
    else:
        optimizer_class = OPTIMIZER_MAP[opt]
        
        kwargs = get_applicable_init_kwargs(optimizer_class, config)
        
        return optimizer_class(parameters, **kwargs)
        


class PytorchBasicTrainer:
    def __init__(self, model, data_loader, loss, optimizer, config=None, callbacks=None):
        self._model = model
        self._loss = loss
        self._data_loader = data_loader
        self._optimizer = optimizer
        self._config = config
        self._callbacks = callbacks or []

    def train_step(self, data, target):
        data, target = data.to('cuda:0'), target.to('cuda:0')

        self._optimizer.zero_grad()
        output = self._model(data)
        loss = self._loss(output, target)
        loss.backward()
        self._optimizer.step()
        
        return loss
        
    def train_epoch(self):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(self._data_loader):
            loss = self.train_step(data, target)
            total_loss += loss.item()
            
            # TODO: refactor
            if batch_idx % 10 == 0:
                print('batch {} - loss {}'.format(batch_idx, total_loss))
                
        return total_loss

    
    def learn(self):
        # move to gpu
        # model = self._model.move_to_gpu(cuda_device=0)
        # TODO: migrate from model move_to_gpu to here?
        
        config_keys = [CONFIG_NUM_EPOCHS, ]
        parsed = parse_config(self._config, config_keys)
        (num_epochs,) = parsed
        
        self._model.train()
        for epoch in range(num_epochs):
            total_loss = self.train_epoch()
            
            # TODO: refactor
            if epoch % 100 == 0:
                print('epoch {} - loss {}'.format(epoch, total_loss))





class TorchVisionModelWrapper:
    disk_size = None  # optional
    
    def __init__(self, model_class):
        self._init_kwargs = {'model_class': model_class}
        self._model = model_class()
        
    def save(self, model_dir, model_file_name='model.th'):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_path = os.path.join(model_dir, model_file_name)
        with open(model_path, 'wb') as f:
            torch.save(self._model.state_dict(), f)
    
    def load(self, model_dir, model_file_name='model.th'):
        model_path = os.path.join(model_dir, model_file_name)
        with open(model_path, 'rb') as f:
            self._model.load_state_dict(torch.load(f))
        return model
        
    def learn(self, data_loader, loss, optimizer, config=None, callbacks=None):
        trainer = PytorchBasicTrainer(model=_model, loss=loss, data_loader=data_loader, 
                                      optimizer=optimizer, config=config, callbacks=callbacks)
        trainer.learn()
        
    # optional
    def download_pretrained_weights(self): pass
    def setup_finetune(self): pass

    
class TorchVisionBaseModelWrapper:
    CLASS_NAME = None
    CLASS_ARGS = []
    CLASS_KWARGS = {}
    
    def __init__(self, *args, **kwargs):
        self._model = self.CLASS_NAME(*args, **kwargs)
    
    def save(self, model_dir, model_file_name='model.th'):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_path = os.path.join(model_dir, model_file_name)
        with open(model_path, 'wb') as f:
            torch.save(self._model.state_dict(), f)
    
    def load(self, model_dir, model_file_name='model.th'):
        model_path = os.path.join(model_dir, model_file_name)
        with open(model_path, 'rb') as f:
            self._model.load_state_dict(torch.load(f))
        
    def learn(self, data_loader, loss, optimizer, config=None, callbacks=None):
        self.move_to_gpu(cuda_device=config.get('cuda_device', -1))
        trainer = PytorchBasicTrainer(model=_model, loss=loss, data_loader=data_loader, 
                                      optimizer=optimizer, config=config, callbacks=callbacks)
        trainer.learn()
        
    def move_to_gpu(self, cuda_device=0):
        print('moving to cuda_device ', cuda_device)
        if cuda_device > -1:
            self._model = self._model.cuda(cuda_device)
        print('moved to cuda_device ', cuda_device)
        
    # optional
    def download_pretrained_weights(self):
        raise NotImplementedError()

    def setup_finetune(self):
        raise NotImplementedError()

class TorchVisionAlexNetModelWrapper(TorchVisionBaseModelWrapper):
    CLASS_NAME = torchvision.models.AlexNet
    CLASS_ARGS = []
    CLASS_KWARGS = {'num_classes': 1000}
        
    def learn(self, data_loader, optimizer='adam', config=None, callbacks=None):
        optimizer = get_optimizer(optimizer, self._model.parameters(), config)
        loss = self.batch_loss()
        trainer = PytorchBasicTrainer(model=self._model, loss=loss, data_loader=data_loader, 
                                      optimizer=optimizer, config=config, callbacks=callbacks)
        trainer.learn()
        
    # optional
    def download_and_load_pretrained_weights(self, model_dir, progress=True):
        # TODO: use model_dir?
        del self._model
        model = torchvision.models.alexnet(pretrained=True, progress=progress)
        self._model = model

    def batch_loss(self):
        return torch.nn.CrossEntropyLoss()
        
    def setup_finetune(self): pass


def main():
    alex_net = TorchVisionAlexNetModelWrapper(num_classes=10)

    mnist = TorchVisionDatasetWrapper(torchvision.datasets.mnist.MNIST, 'mnist', download=True)
    mnist.download_and_prepare()

    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(), 
        transforms.Lambda(lambda x: torch.cat([x, x, x], 0))
    ])

    def get_one_hot(num_classes):
        def to_one_hot(num):
            y_onehot = torch.zeros((num_classes,))
            y_onehot[num] = 1.0
            return y_onehot
        
        return to_one_hot
        

    target_transform = None # transforms.Lambda(get_one_hot(10))

    mnist = TorchVisionDatasetWrapper(torchvision.datasets.mnist.MNIST, 'mnist', download=True,
                                      transform=transform, target_transform=target_transform,)
    mnist.download_and_prepare()

    data_loader = mnist.as_pytorch_dataloader(batch_size=32)
    config = {
        'lr': 0.001,
        'num_epochs': 1
    }
    alex_net.move_to_gpu(cuda_device=0)
    alex_net.learn(data_loader=data_loader, config=config)

main()