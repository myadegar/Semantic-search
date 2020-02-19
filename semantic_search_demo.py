from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors
from argparse import ArgumentParser


class Semantic():
    def __init__(self, glove_file):
        tmp_file = get_tmpfile("test_word2vec.txt")
        glove2word2vec(glove_file, tmp_file)
        self.model = KeyedVectors.load_word2vec_format(tmp_file)

    def __call__(self, word):
        semantic_words = self.model.most_similar(word)
        semantic_words = [word for word,_ in semantic_words]
        return semantic_words



def build_parser():
    par = ArgumentParser()
    par.add_argument('--search_word', type=str,
                     dest='search_word', help='Select a search word',
                     default='teacher')
    par.add_argument('--glove_model_path', type=str,
                     dest='glove_model_path', help='Relative destination to glove model path',
                     default="models/glove.6B.100d.txt")
    return par



if __name__ == '__main__':
    parser = build_parser()
    options = parser.parse_args()
    glove_model_path = options.glove_model_path
    search_word = options.search_word
    semantic = Semantic(glove_model_path)
    semantic_words = semantic(search_word)
    print('Word: ', search_word)
    semantics = ''
    for wrd in semantic_words:
        semantics += (wrd + ', ')
    semantics = semantics.rstrip(', ')
    print('Semantic words: ', semantics)


