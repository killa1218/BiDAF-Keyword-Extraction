import argparse
import json
import os
# data: q, cq, (dq), (pq), y, *x, *cx
# shared: x, cx, (dx), (px), word_counter, char_counter, word2vec
# no metadata
from collections import Counter

from tqdm import tqdm

from .utils import get_word_span, get_word_idx, process_tokens


def main():
    args = get_args()
    prepro(args)


def get_args():
    parser = argparse.ArgumentParser()
    home = os.path.expanduser("~")
    source_dir = "/mnt/dataset/KeyphraseExtraction/ClearData/krapivin/tokenize/nopunc/nostem/"
    target_dir = "data/krapivin"
    glove_dir = os.path.join(home, "data", "glove")
    parser.add_argument('-s', "--source_dir", default=source_dir)
    parser.add_argument('-t', "--target_dir", default=target_dir)
    parser.add_argument('-d', "--debug", action='store_true')
    parser.add_argument('--max_len', default = '', type = str)
    parser.add_argument('--max_num', default = None, type = int)
    parser.add_argument("--train_ratio", default=0.9, type=int)
    parser.add_argument("--glove_corpus", default="6B")
    parser.add_argument("--glove_dir", default=glove_dir)
    parser.add_argument("--glove_vec_size", default=100, type=int)
    parser.add_argument("--mode", default="full", type=str)
    parser.add_argument("--single_path", default="", type=str)
    parser.add_argument("--tokenizer", default="PTB", type=str)
    parser.add_argument("--url", default="vision-server2.corp.ai2", type=str)
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--split", action='store_true')
    # TODO : put more args here
    return parser.parse_args()


def create_all(args):
    out_path = os.path.join(args.source_dir, "all.json")
    if os.path.exists(out_path):
        return
    train_path = os.path.join(args.source_dir, "train.json")
    train_data = json.load(open(train_path, 'r'))
    dev_path = os.path.join(args.source_dir, "valid.json")
    dev_data = json.load(open(dev_path, 'r'))
    train_data.extend(dev_data)
    print("dumping all data ...")
    json.dump(train_data, open(out_path, 'w'))


def prepro(args):
    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)

    if args.max_len:
        postfix = '_' + str(args.max_len)
    else:
        postfix = ''

    if args.mode == 'full':
        prepro_each(args, 'train' + postfix, out_name='train')
        prepro_each(args, 'valid' + postfix, out_name='dev')
        prepro_each(args, 'test' + postfix, out_name='test')
    elif args.mode == 'all':
        create_all(args)
        prepro_each(args, 'valid' + postfix, 0.0, 0.0, out_name='dev')
        prepro_each(args, 'test' + postfix, 0.0, 0.0, out_name='test')
        prepro_each(args, 'all' + postfix, out_name='train')
    elif args.mode == 'single':
        assert len(args.single_path) > 0
        prepro_each(args, "NULL", out_name="single", in_path=args.single_path)
    else:
        prepro_each(args, 'train' + postfix, 0.0, args.train_ratio, out_name='train')
        prepro_each(args, 'valid' + postfix, args.train_ratio, 1.0, out_name='dev')
        prepro_each(args, 'valid' + postfix, out_name='test')


def save(args, data, shared, data_type):
    data_path = os.path.join(args.target_dir, "data_{}.json".format(data_type))
    shared_path = os.path.join(args.target_dir, "shared_{}.json".format(data_type))
    json.dump(data, open(data_path, 'w'))
    json.dump(shared, open(shared_path, 'w'))


# def get_word2vec(args, word_counter):
#     glove_path = os.path.join(args.glove_dir, "glove.{}.{}d.txt".format(args.glove_corpus, args.glove_vec_size))
#     sizes = {'6B': int(4e5), '42B': int(1.9e6), '840B': int(2.2e6), '2B': int(1.2e6)}
#     total = sizes[args.glove_corpus]
#     word2vec_dict = {}
#     with open(glove_path, 'r', encoding='utf-8') as fh:
#         for line in tqdm(fh, total=total):
#             array = line.lstrip().rstrip().split(" ")
#             word = array[0]
#             vector = list(map(float, array[1:]))
#             if word in word_counter:
#                 word2vec_dict[word] = vector
#             elif word.capitalize() in word_counter:
#                 word2vec_dict[word.capitalize()] = vector
#             elif word.lower() in word_counter:
#                 word2vec_dict[word.lower()] = vector
#             elif word.upper() in word_counter:
#                 word2vec_dict[word.upper()] = vector
#
#     print("{}/{} of word vocab have corresponding vectors in {}".format(len(word2vec_dict), len(word_counter), glove_path))
#     return word2vec_dict


def prepro_each(args, data_type, start_ratio=0.0, stop_ratio=1.0, out_name="default", in_path=None):
    if args.tokenizer == "PTB":
        import nltk
        sent_tokenize = nltk.sent_tokenize
        def word_tokenize(tokens):
            return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]
    elif args.tokenizer == 'Stanford':
        from my.corenlp_interface import CoreNLPInterface
        interface = CoreNLPInterface(args.url, args.port)
        sent_tokenize = interface.split_doc
        word_tokenize = interface.split_sent
    else:
        raise Exception()

    if not args.split:
        sent_tokenize = lambda para: [para]

    source_path = in_path or os.path.join(args.source_dir, "{}.json".format(data_type))
    source_data = json.load(open(source_path, 'r'))

    y, rx, rcx, ids, idxs = [], [], [], [], []
    cy = []
    x, cx = [], []
    answerss = []
    p = []
    word_counter, char_counter, lower_word_counter = Counter(), Counter(), Counter()
    start_ai = int(round(len(source_data) * start_ratio))
    stop_ai = int(round(len(source_data) * stop_ratio))
    for ai, article in enumerate(tqdm(source_data[start_ai:stop_ai])):
        if args.max_num and ai == args.max_num:
            break
        xp, cxp = [], []
        pp = []
        x.append(xp) # x: [[[[
        cx.append(cxp) # cx: [[[[[
        p.append(pp) # p: [["context string", ""]]

        context = article['passage']

        if isinstance(context, list):
            xi = [context]
            c = ' '.join(context)
        else:
            context = context.replace("''", '" ')
            context = context.replace("``", '" ')
            c = context
            xi = list(map(word_tokenize, sent_tokenize(context)))

        xi = [process_tokens(tokens) for tokens in xi]  # process tokens
        xi = [[xijk for xijk in xij if xijk != ''] for xij in xi]
        # given xi, add chars
        cxi = [[list(xijk) for xijk in xij] for xij in xi]
        xp.append(xi)
        cxp.append(cxi)
        pp.append(c)

        for xij in xi:
            for xijk in xij:
                # word_counter[xijk] += 1
                # lower_word_counter[xijk.lower()] += 1
                for xijkl in xijk:
                    char_counter[xijkl] += 1

        pi = 0
        rxi = [ai, pi]
        assert len(x) - 1 == ai
        assert len(x[ai]) - 1 == pi

        # get words
        yi = []
        cyi = []
        answers = []
        for answer in article['keyphrase']:
            answer_text = answer['text'].strip()
            answers.append(answer_text)
            # TODO : put some function that gives word_start, word_stop here
            yi0 = (0, answer['start_position']) # keyphrase first word word index
            yi1 = (0, answer['end_position']) # keyphrase last word word index

            assert len(xi[yi0[0]]) > yi0[1]
            assert len(xi[yi1[0]]) >= yi1[1]

            cyi0 = 0 # 应该为0
            cyi1 = len(answer_text.split(' ')[-1]) - 1 # answer 最后一个词的最后一个字母的位置

            yi.append([yi0, yi1])
            cyi.append([cyi0, cyi1])

        y.append(yi) # y: [[[(stc_idx, start_word_idx), (stc_idx, end_word_idx)]]] 左闭右开
        cy.append(cyi) # cy: [[[
        rx.append(rxi) # 用于指定当前位置answer对应share中的哪个context  rx: [[
        rcx.append(rxi) # 作用同上 rcx: [[
        ids.append(article['id']) # ids: [
        # ids.append(md5(c.encode('utf8')).hexdigest()[-24:]) # ids: [
        idxs.append(len(idxs)) # idxs: [
        answerss.append(answers) # answer strings answerss: [["answer string", ""]]

        if args.debug:
            break

    with open(os.path.join(args.source_dir, 'vocab.json')) as f:
        vocab = json.load(f)
        # word2vec_dict = vocab['word2vec']
        # lower_word2vec_dict = word2vec_dict
        word_counter = vocab['word2count']
        lower_word_counter = word_counter

    # word2vec_dict = get_word2vec(args, word_counter)
    # lower_word2vec_dict = get_word2vec(args, lower_word_counter)

    # add context here
    data = {'y': y, '*x': rx, '*cx': rcx, 'cy': cy,
            'idxs': idxs, 'ids': ids, 'answerss': answerss, '*p': rx}
    shared = {'x': x, 'cx': cx, 'p': p,
              'word_counter': word_counter,
              'char_counter': char_counter,
              'lower_word_counter': lower_word_counter,
              # 'word2vec': word2vec_dict,
              # 'lower_word2vec': lower_word2vec_dict,
              'word2idx': vocab['word2idx'],
              'idx2vec': vocab['idx2vec']
              }

    print("saving ...")
    save(args, data, shared, out_name)



if __name__ == "__main__":
    main()
