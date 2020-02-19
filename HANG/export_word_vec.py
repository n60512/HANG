import torch
from utils.preprocessing import Preprocess
from utils import options

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
opt = options.GatherOptions().parse()

def StoreWordSemantic(model_path, voc, fname, dim=300):
    
    model = torch.load(model_path)

    with open(fname, 'a') as _file:
        _file.write('{} {}\n'.format(len(voc.index2word), dim))

    for index in voc.index2word:
        with open(fname, 'a') as _file:
            tmpStrls = model.embedding(torch.tensor(index).to(device)).tolist()
            tmpStr = ''
            for val in tmpStrls:
                tmpStr = tmpStr + str(val) + ' '
            _file.write('{} {}\n'.format(voc.index2word[index], tmpStr))
        

if __name__ == "__main__":
    

    pre_work = Preprocess()

    res, itemObj, userObj = pre_work.loadData( havingCount=opt.having_interactions, 
        LIMIT=2000, sqlfile=opt.sqlfile, testing=False, table= opt.selectTable)  # for clothing.

    # Generate voc & User information
    voc, USER = pre_work.Generate_Voc_User(res, havingCount=opt.having_interactions, 
        limit_user=99999)

    # model_path = 'HNAE/log/origin/20200109_10_47_interaction@15_review@4/Model/IntraGRU_idx0_epoch2'
    # fname = 'HNAE/data/trained_wordVector/interaction@15_review@4_nopre.vec'
    model_path = 'HNAE/log/origin/20200109_10_54_interaction@15_review@4_PretrainWV/Model/IntraGRU_idx0_epoch5'
    fname = 'HNAE/data/trained_wordVector/interaction@15_review@4_withpre.vec'

    StoreWordSemantic(model_path, voc, fname)