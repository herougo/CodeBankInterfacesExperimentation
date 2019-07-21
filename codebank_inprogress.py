def AbstractDatasetProblemGuideline:
  urls = {} # dict mapping file_name to url
  disk_size = None
  description = None
  citation = None

  def download_and_prepare(self, dir_path):
    raise NotImplementedError()
  

class DatasetProblemGuideline(AbstractDatasetProblemGuideline):
  n_examples = None
  n_train_examples = None
  n_val_examples = None
  n_test_examples = None

  def __init__(self, transformer=None):
    self.transformer = transformer

  # recommended
  def __getitem__(self, i):
    # get ith example (should be implemented if possible)
    raise NotImplementedError()
  def iter_raw_examples(self):
    # read directly from the files
	# yields an example (tuple of python variables) (e.g. (question, answer, background))
    raise NotImplementedError()
  def __iter__(self): raise NotImplementedError()
  @property
  def shape(self): raise NotImplementedError()
  
  # recommended numpy logic
  # doesn't rely on pytorch or tensorflow and can be easily adapted for tensorflow or pytorch
  def preprocess_example_numpy(self, example):  raise NotImplementedError()
  def iter_numpy_examples(self): raise NotImplementedError()

  # specific to pytorch
  def preprocess_example_pytorch(self, example): raise NotImplementedError()
  def as_pytorch_dataset(self): raise NotImplementedError()
  def as_pytorch_dataloader(self): raise NotImplementedError()
  
  # specific to tensorflow
  def preprocess_example_tf(self, example): raise NotImplementedError()
  def as_tf_dataset(self): raise NotImplementedError()
  def as_tf_iterator(self): raise NotImplementedError()
	
'''
  def iter_tf_batches(self):
    # iter over dictionary batches of tensorflow tensors
    raise NotImplementedError()

  def iter_pytorch_batches(self):
    # iter over dictionary batches of pytorch tensors
    raise NotImplementedError()
'''

def model_sanity_check(model):
  """
  Checks:
  - has recommended attributes
  - has recommended functions
  - can download pretrained weights properly
  - can make a prediction properly
  - can save and load properly
  Calculates
  - model disk size

  Since multiple errors can happen, it makes all these checks,
  then throws one big error at the end (as opposed to only sending one
  error at a time).
  """
  errors = []
  warns = []

  assert len(errors) == 0, errors

def trainer_sanity_check(trainer):
  """
  Checks:
  - has recommended attributes
  - has recommended functions
  - can checkpoint
  - can make a prediction properly
  - can save and load properly
  - can run through one epoch
  - pytorch-specific
    - support pytorch.data.DataLoader
  - performance:
    - data loading isn't a bottle-neck
  Calculates:
  - run-time
  - expected train-time values
  - GPU-memory usage

  Since multiple errors can happen, it makes all these checks,
  then throws one big error at the end (as opposed to only sending one
  error at a time).
  """
  errors = []
  warns = []

  assert len(errors) == 0, errors