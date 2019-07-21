from typing import Iterator, List, Dict
import torch
import torch.optim as optim
import numpy as np
from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common.file_utils import cached_path
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.predictors import SentenceTaggerPredictor

torch.manual_seed(1)
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from enum import Enum

ALL = -1
TRAIN = 0
VALIDATION = 1
TEST = 2


class _PosReaderFileIds(Enum):
    TRAIN = 'train'
    VALIDATION = 'validation'

_SPLIT_TO_FILE_ID = {
    TRAIN: _PosReaderFileIds.TRAIN,
    VALIDATION: _PosReaderFileIds.VALIDATION
}
_FILE_ID_TO_SPLIT = {
    _PosReaderFileIds.TRAIN: TRAIN,
    _PosReaderFileIds.VALIDATION: VALIDATION
}

class _LazyIterableDataset(Dataset):
    def __init__(self, iterable, n_examples):
        self.iterable = iterable
        self.n_examples = n_examples
        # for some reason IterableDataset isn't in ie "pip install torch"
        self.cache = [example for example in self]
        
    def __getitem__(self, i):
        return self.cache[i]

    def __iter__(self):
        for example in self.iterable:
            yield example
    
    def __len__(self):
        return self.n_examples

def iterable_to_pytorch_dataset(iterable, n_examples):
    return _LazyIterableDataset(iterable, n_examples)


class PosDatasetReader(DatasetReader):
    """
    DatasetReader for PoS tagging data, one sentence per line, like

        The###DET dog###NN ate###V the###DET apple###NN
    """
    n_examples = {
        ALL: 4,
        TRAIN: 2,
        VALIDATION: 2,
        TEST: None
    }

    urls = {
        _PosReaderFileIds.TRAIN: ('https://raw.githubusercontent.com/allenai/allennlp'
                        '/master/tutorials/tagger/training.txt'),
        _PosReaderFileIds.VALIDATION: ('https://raw.githubusercontent.com/allenai/allennlp'
                          '/master/tutorials/tagger/validation.txt')
    }
    description = None
    citation = None

    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

        # extra code bank attributes
        self.data_dir = None
        self.data_file_paths = {}


    # allennlp-specific
    def text_to_instance(self, tokens: List[Token], tags: List[str] = None) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"sentence": sentence_field}

        if tags:
            label_field = SequenceLabelField(labels=tags, sequence_field=sentence_field)
            fields["labels"] = label_field

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as f:
            for line in f:
                pairs = line.strip().split()
                sentence, tags = zip(*(pair.split("###") for pair in pairs))
                yield self.text_to_instance([Token(word) for word in sentence], tags)

    # extra code bank functions
    def download_and_prepare(self, dir_path=None):
        train_path = cached_path(self.urls[_PosReaderFileIds.TRAIN], cache_dir=dir_path)
        val_path = cached_path(self.urls[_PosReaderFileIds.VALIDATION], cache_dir=dir_path)
        self.data_dir = dir_path
        self.data_file_paths = {
            TRAIN: train_path,
            VALIDATION: val_path
        }
        return self.data_file_paths

    def iter_raw_examples(self, split=TRAIN):
        ''' Yields like:
          sentence: ('Everybody', 'ate', 'that', 'book')
          tags:     ('NN', 'V', 'DET', 'NN')
        '''
        # copy-pasted to avoid allennlp interface conflicts
        file_path = self.data_file_paths[split]
        with open(file_path) as f:
            for line in f:
                pairs = line.strip().split()
                sentence, tags = zip(*(pair.split("###") for pair in pairs))
                yield (sentence, tags)

    # recommended
    def __getitem__(self, i):
        raise NotImplementedError()

    # specific to pytorch
    def iter_pytorch_examples(self, split=TRAIN):
        for example in self.iter_raw_examples(split=split):
            yield self.preprocess_example_pytorch(example, split=split)

    def preprocess_example_pytorch(self, example, split=TRAIN):
        sentence, tags = example
        instance = self.text_to_instance([Token(word) for word in sentence], tags)
        instance.index_fields(self._allennlp_vocab)
        tensor_dict = instance.as_tensor_dict()
        sentence_tensor = tensor_dict['sentence']['tokens']
        tag_labels_tensor = tensor_dict['labels']
        return (sentence_tensor, tag_labels_tensor)

    def as_pytorch_dataset(self, split=TRAIN):
        return iterable_to_pytorch_dataset(self.iter_pytorch_examples(split=split), 
                                           self.n_examples[split])

    def as_pytorch_dataloader(self, split=TRAIN, **kwargs):
        dataset = self.as_pytorch_dataset(split=TRAIN)
        return DataLoader(dataset, **kwargs)

    # recommended numpy logic (depends on pytorch)
    def preprocess_example_numpy(self, example):
        tensor_tuple = self.preprocess_example_pytorch(example)
        return [tensor.numpy() for tensor in tensor_tuple]

    def iter_numpy_examples(self):
        for example in self.iter_raw_examples(split=split):
            yield self.preprocess_example_numpy(example)

    # added as needed
    def build_allennlp_vocab(self, splits=None):
        if splits is None:
            splits = [TRAIN, VALIDATION]
        iterator = [instance
                    for split in splits
                    for instance in self.read(self.data_file_paths[split])]
        vocab = Vocabulary.from_instances(iterator)
        return vocab

    def set_allennlp_vocab(self, allennlp_vocab):
        self._allennlp_vocab = allennlp_vocab


class LstmTagger(Model):
    example_disk_size = None
  
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.hidden2tag = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                          out_features=vocab.get_vocab_size('labels'))
        self.accuracy = CategoricalAccuracy()
    def forward(self,
                sentence: Dict[str, torch.Tensor],
                labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(sentence)
        embeddings = self.word_embeddings(sentence)
        encoder_out = self.encoder(embeddings, mask)
        tag_logits = self.hidden2tag(encoder_out)
        output = {"tag_logits": tag_logits}
        if labels is not None:
            self.accuracy(tag_logits, labels, mask)
            output["loss"] = sequence_cross_entropy_with_logits(tag_logits, labels, mask)

        return output
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}
      
    # CODE BANK extra functions
    def move_to_gpu(self, cuda_device=-1):
        if cuda_device > -1:
            self.cuda(cuda_device)
    
    def save(self, model_dir, model_file_name='model.th', vocab_file_name='vocabulary'):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_path = os.path.join(model_dir, model_file_name)
        vocab_path = os.path.join(model_dir, vocab_file_name)
        with open(model_path, 'wb') as f:
            torch.save(model.state_dict(), f)
        vocab.save_to_files(vocab_path)
    
    @classmethod
    def load_from(self, model_dir, model_file_name='model.th', vocab_file_name='vocabulary'):
        model_path = os.path.join(model_dir, model_file_name)
        vocab_path = os.path.join(model_dir, vocab_file_name)
        vocab = Vocabulary.from_files(vocab_path)
        model = LstmTagger(word_embeddings, lstm, vocab)
        with open(model_path, 'rb') as f:
            model.load_state_dict(torch.load(f))
        return model
    
    def download_pretrained_weights(self):
        raise NotImplementedError()


EMBEDDING_DIM = 6
HIDDEN_DIM = 6

EMBEDDING_DIM = 6
HIDDEN_DIM = 6

reader = PosDatasetReader()
cached_paths = reader.download_and_prepare()

train_dataset = reader.read(cached_paths[TRAIN])
validation_dataset = reader.read(cached_paths[VALIDATION])
vocab = Vocabulary.from_instances(train_dataset + validation_dataset)
token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                            embedding_dim=EMBEDDING_DIM)
word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))
model = LstmTagger(word_embeddings, lstm, vocab)
if torch.cuda.is_available():
    cuda_device = 0
    model = model.move_to_gpu(cuda_device)
else:
    cuda_device = -1
optimizer = optim.SGD(model.parameters(), lr=0.1)
iterator = BucketIterator(batch_size=2, sorting_keys=[("sentence", "num_tokens")])
iterator.index_with(vocab)

trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_dataset,
                  validation_dataset=validation_dataset,
                  patience=10,
                  num_epochs=1000,
                  cuda_device=cuda_device)

predictor = SentenceTaggerPredictor(model, dataset_reader=reader)
tag_logits = predictor.predict("The dog ate the apple")['tag_logits']
tag_ids = np.argmax(tag_logits, axis=-1)
print([model.vocab.get_token_from_index(i, 'labels') for i in tag_ids])

model.save('/tmp/', model_file_name='model.th', vocab_file_name='vocabulary')
model.load('/tmp', model_file_name='model.th', vocab_file_name='vocabulary')
model.move_to_gpu(cuda_device=cuda_device)

predictor2 = SentenceTaggerPredictor(model2, dataset_reader=reader)
tag_logits2 = predictor2.predict("The dog ate the apple")['tag_logits']
np.testing.assert_array_almost_equal(tag_logits2, tag_logits)



# Use pytorch DataLoader
reader = PosDatasetReader()
cached_paths = reader.download_and_prepare()

train_dataset = reader.read(cached_paths[TRAIN])
validation_dataset = reader.read(cached_paths[VALIDATION])
vocab = Vocabulary.from_instances(train_dataset + validation_dataset)
token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                            embedding_dim=EMBEDDING_DIM)
word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

vocab = reader.build_allennlp_vocab()
reader.set_allennlp_vocab(vocab)
train_dataloader = reader.as_pytorch_dataloader(split=TRAIN, batch_size=1)
val_dataloader = reader.as_pytorch_dataloader(split=VALIDATION, batch_size=1)

lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))
model2 = LstmTagger(word_embeddings, lstm, vocab)
model2.move_to_gpu(cuda_device=0)
optimizer = optim.SGD(model2.parameters(), lr=0.1)

model.train()
for epoch in range(1000):
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_dataloader):
        data, target = data.to('cuda:0'), target.to('cuda:0')

        data = {'tokens': data} # handling allennlp integration
        
        optimizer.zero_grad()
        output = model2(data, target)
        loss = output['loss']
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    if epoch % 100 == 0:
        print('epoch {} - loss {}'.format(epoch, total_loss))

predictor3 = SentenceTaggerPredictor(model2, dataset_reader=reader)
tag_logits3 = predictor3.predict("The dog ate the apple")['tag_logits']
np.testing.assert_array_almost_equal(tag_logits3, tag_logits3)





''' Remaining Issues

Major Improvements for the Future
- better way to do iterable to dataset
- find way to clean up mapping split to file id and to url/cached path
- handling vocabularies (better than set_allennlp_vocab)

Future Decisions
- What exactly should preprocess_example_pytorch return? dict of tensors? nested dict?
- default implementations for allennlp wrappers (e.g. preprocess_example_pytorch)

Allennlp look into
- whether allennlp supports cached iterators
- how allennlp supports multiprocessing for dataloaders

Major Problems
- if someone just wants the allennlp DataReader without the extra stuff, it is 
hard to refactor that way
- does not handle padding

Minor Problems
- Depends on allennlp vocab
- missing shape property
- record cuda_device as an attribute
- This example doesn't handle padding

Learned
- pytorch DataIterator returns non-dict's (bad)


'''