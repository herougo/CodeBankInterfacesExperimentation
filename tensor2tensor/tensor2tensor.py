import tensorflow as tf
from tensor2tensor.data_generators.celeba import ImageCeleba, Img2imgTransformer
import os

DATA_DIR = 

TRAIN = 0
VALIDATION = 1
TEST = 2

_SPLIT_TO_MODE = {
    TRAIN: tf.estimator.ModeKeys.TRAIN
}

class Tensor2TensorProblemWrapper:
  problem = Img2imgCeleba()
  urls = {}
  disk_size = None
  description = None
  citation = None

  n_examples = None
  n_train_examples = None
  n_val_examples = None
  n_test_examples = None

  def download_and_prepare(self, dir_path, temp_dir=None):
    if not os.path.exists(dir_path):
      os.makedirs(dir_path)
    self.problem.generate_data(DATA_DIR, TMP_DIR)
    self.data_dir = self.data_dir

  def as_tf_dataset(self, split=TRAIN, **kwargs):
    return self.problem.dataset(mode=_SPLIT_TO_MODE[split], 
                                data_dir=self.dir_path, **kwargs)

  # recommended numpy logic
  def iter_numpy_examples(self, split=TRAIN): raise NotImplementedError()
    dataset = self.as_tf_dataset(split=split)
    features = tf.compat.v1.data.make_one_shot_iterator(batched_dataset).get_next()
    try:
      while True:
        d = sess.run(features)  # {'inputs': ..., 'targets': ...}
        image, label = features['inputs'], features['targets']
        yield image, label
    except tf.errors.OutOfRangeError:
      pass

  # specific to pytorch
  def preprocess_example_pytorch(self, example): raise NotImplementedError()
  def as_pytorch_dataset(self): raise NotImplementedError()
  def as_pytorch_dataloader(self): raise NotImplementedError()




'''
Questions:
- can you import models and train them?


'''