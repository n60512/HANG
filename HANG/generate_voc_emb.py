from gensim.models.wrappers import FastText
from gensim.models import KeyedVectors
from utils.preprocessing import Preprocess
import io
import torch

def writeVoc(sqlfile= 'HANG/SQL/elec_select_all.sql', selectTable = 'clothing_'):

    res, itemObj, userObj = pre_work.load_data(sqlfile=sqlfile, testing=False, table= selectTable)  
    print('end sql.')
    # Generate voc 
    # voc, _,_ = pre_work.generate_candidate_voc(res, generate_voc=True, net_type = 'item_base')

    voc = pre_work.generate_voc(res)
    fname = '{}ALL_Voc.txt'.format(selectTable)

    with open('HANG/data/' + fname, 'w') as file:
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

    if True:
        writeVoc(selectTable = 'elec_')

    # if True:
    #     with open('HNAE/data/toys_ALL_Voc.txt', 'r') as file:
    #         content = file.read()
    #     words = content.split(',')


    # if True:
    #     fname = '/home/kdd2080ti/Documents/Sean/RecommendationSystem/PretrainWord/wiki-news-300d-1M.vec'
    #     model = KeyedVectors.load_word2vec_format(fname, binary=False)

    #     print("Loading complete.")

    # if True:
    #     StoreWordSemantic(words, 300, "HNAE/data/toys_festtext_subEmb.vec")