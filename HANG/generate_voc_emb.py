from gensim.models.wrappers import FastText
from gensim.models import KeyedVectors
from utils.preprocessing import Preprocess
import io
import torch

def writeVoc(sqlfile= 'HNAE/SQL/toys_selectALL.sql', selectTable = 'clothing_'):

    res, itemObj, userObj = pre_work.loadData( sqlfile=sqlfile, testing=False, table= selectTable)  
    # Generate voc 
    voc = pre_work.Generate_Voc(res)
    fname = 'toys_ALL_Voc.txt'

    with open('HNAE/data/' + fname, 'w') as file:
        for word in voc.word2index:
            file.write('{},'.format(word))


def StoreWordSemantic(words, dim, fname):
    
    with open(fname, 'a') as _file:
        _file.write('{} {}\n'.format(len(words), dim))

    with open(fname, 'a') as _file:
        for word in words:

            try:
                wordVec = model.wv[word]
            except KeyError as msg:
                wordVec = torch.randn(dim)
                           
            tmpStr = ''
            # Each dim val
            for val in wordVec:
                tmpStr = tmpStr + str(float(val)) + ' '        
            
            _file.write('{} {}\n'.format(word, tmpStr))
        


# %%
# from gensim.models import KeyedVectors

# filename = 'HNAE/data/toys_festtext_subEmb.vec'
# model_test = KeyedVectors.load_word2vec_format(filename, binary=False)

# model_test.most_similar('great', topn=5)

if __name__ == "__main__":
    
    pre_work = Preprocess(2562340)

    if not True:
        writeVoc(selectTable = 'toys_')

    if True:
        with open('HNAE/data/toys_ALL_Voc.txt', 'r') as file:
            content = file.read()
        words = content.split(',')


    if True:
        fname = '/home/kdd2080ti/Documents/Sean/RecommendationSystem/PretrainWord/wiki-news-300d-1M.vec'
        model = KeyedVectors.load_word2vec_format(fname, binary=False)

        print("Loading complete.")

    if True:
        StoreWordSemantic(words, 300, "HNAE/data/toys_festtext_subEmb.vec")