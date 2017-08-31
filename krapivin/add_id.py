import os
import json
from tqdm import tqdm

gloveFile = 'data/glove'
vocabFile = 'data/krapivin/origin/count.json'
dataDir = 'data/krapivin/origin'
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

    print("{}/{} of word vocab have corresponding vectors in {}".format(len(word2vec_dict), 0, glove_path))
    return word2vec_dict

w2v = get_word2vec(glove_dir = gloveFile)

# Remove case and rebuild vocab
with open(vocabFile, 'w') as f:
    vocab = json.load(f)
    idx2word = vocab['idx2word']
    word2count = vocab['word2count']

    idx2vec = [[0.0] * dim, w2v['UNK']]
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
                    idx2vec.append(w2v['UNK'])

    vocab['idx2word'] = newIdx2word
    vocab['word2idx'] = newWord2idx
    vocab['word2count'] = newWord2count
    vocab['idx2vec'] = idx2vec

    json.dump(vocab, f)


dataFileType = [
    'train',
    'test',
    'valid'
]
# Process data files
for dataType in dataFileType:
    filePath = os.path.join(dataDir, dataType + '_tf_idf.json')

    with open(filePath, 'w') as f:
        data = json.load(f)['test']

        for d in data:

