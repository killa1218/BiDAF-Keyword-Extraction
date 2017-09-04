import os
import json
import numpy as np
from tqdm import tqdm
from hashlib import md5

dataset = 'ke20k'


gloveFile = '/home/killa/data/glove'
vocabFile = '/mnt/dataset/KeyphraseExtraction/ClearData/' + dataset + '/tokenize/nopunc/nostem/count.json'
newVocabFile = '/mnt/dataset/KeyphraseExtraction/ClearData/' + dataset + '/tokenize/nopunc/nostem/vocab.json'
dataDir = '/mnt/dataset/KeyphraseExtraction/ClearData/' + dataset + '/tokenize/nopunc/nostem/'
dim = 100

def get_word2vec(glove_dir):
    glove_path = os.path.join(glove_dir, "glove.6B.{}d.txt".format(dim))
    total = int(4e5)
    word2vec_dict = {}
    with open(glove_path, 'r', encoding='utf-8') as fh:
        for line in tqdm(fh, total=total):
            array = line.lstrip().rstrip().split(" ")
            word = array[0]
            vector = list(map(float, array[1:]))
            word2vec_dict[word] = vector

    return word2vec_dict

w2v = get_word2vec(glove_dir = gloveFile)
UNK = list(np.random.random([dim]))

# Remove case and rebuild vocab
with open(vocabFile, 'r') as f:
    vocab = json.load(f)
    idx2word = vocab['idx2word']
    word2count = vocab['word2count']

    idx2vec = [[0.0] * dim, UNK]
    newIdx2word = ['ALLZERO', 'UNK']
    newWord2idx = {'ALLZERO': 0, 'UNK': 1}
    newWord2count = {}

    for word in idx2word:
        if len(word) != 0:
            lowerWord = word.lower()
            newWord2count[lowerWord] = newWord2count.get(lowerWord, 0) + word2count.get(word, 0)

            if lowerWord not in newWord2idx:
                newWord2idx[lowerWord] = len(newIdx2word)
                newIdx2word.append(lowerWord)

                if lowerWord in w2v:
                    idx2vec.append(w2v[lowerWord])
                else:
                    idx2vec.append(UNK)

    vocab['idx2word'] = newIdx2word
    vocab['word2idx'] = newWord2idx
    vocab['word2count'] = newWord2count
    vocab['idx2vec'] = idx2vec

    with open(os.path.join(dataDir, 'vocab.json'), 'w') as ff:
        json.dump(vocab, ff)

dataFileType = [
    'train',
    'test',
    'valid'
]
# Process data files
for dataType in dataFileType:
    filePath = os.path.join(dataDir, dataType + '_all.json')

    with open(filePath, 'r') as f:
        data = json.load(f)

        for d in data:
            psg = d['abstract']
            d['id'] = md5(' '.join(psg).encode('utf8')).hexdigest()
            d.pop('abstract')
            d['passage'] = psg

            for kp in d['keyphrase']:
                kp['end_position'] = kp['end_position'] + 1

        with open(os.path.join(dataDir, dataType + '.json'), 'w') as ff:
            json.dump(data, ff)
