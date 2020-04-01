from utils.preprocessing import Preprocess
import argparse
import random
import pickle

def wpickle(fpath, fname, _dict):
    # pickle a variable to a file
    _file = open(
        '{}/review_sparsity_{}.pickle'.format(fpath, fname), 
        'wb'
        )
    pickle.dump(_dict, _file)
    _file.close()    

def _store(data_preprocess, opt, testing=False, select_review=-1, write_can2item=False):

    item_net_sql = "HANG/SQL/cloth_interaction@6_itembase_rm_sw.sql"
    base_model_net_type = 'item_base'
    correspond_model_net_type = 'user_base'

    res, itemObj, userObj = data_preprocess.load_data(
        sqlfile=item_net_sql, 
        testing=testing,
        table= opt.selectTable, 
        rand_seed=opt.train_test_rand_seed
        )  # for clothing.

    # Generate voc & (User or Item) information , CANDIDATE could be USER or ITEM
    ITEM, candiate2index = data_preprocess.generate_candidate_voc(
        res, 
        having_interaction=opt.having_interactions, 
        generate_voc=False,
        net_type = base_model_net_type
        )


    """
    Generate user-net training data.
    """
    num_of_reviews_unet = 4

    user_base_sql = R'HANG/SQL/cloth_candidate_asin.sql'
    res, itemObj, userObj = data_preprocess.load_data(
        sqlfile=user_base_sql, 
        testing=testing, 
        table= opt.selectTable, 
        rand_seed=opt.train_test_rand_seed
        )

    # Generate voc & (User or Item) information , CANDIDATE could be USER or ITEM
    USER, uid2index = data_preprocess.generate_candidate_voc(
        res, 
        having_interaction=opt.having_interactions, 
        net_type = correspond_model_net_type,
        generate_voc=False
        )


    # Export the `Consumer's history` through chosing number of `candidate`
    chosing_num_of_candidate = opt.num_of_reviews
    ITEM_CONSUMER = list()
    for _item in ITEM:
        candidate_uid = _item.this_reviewerID[chosing_num_of_candidate]
        user_index = uid2index[candidate_uid]   

        # Append the user which is chosen into consumer list
        ITEM_CONSUMER.append(USER[user_index])   

    ITEM_CONSUMER

    if(select_review == -1):
        # random chose
        drop_count = 2
        can2sparsity = dict()
        for index, _candidate in enumerate(ITEM_CONSUMER):
            can2sparsity[_candidate.reviewerID] = [1,1,1,1]

            for val in range(random.randint(0,drop_count)):
                select = random.randint(0,3) 
                can2sparsity[_candidate.reviewerID][select] = 0

    else:
        # select target review
        can2sparsity = dict()
        for index, _candidate in enumerate(ITEM_CONSUMER):
            can2sparsity[_candidate.reviewerID] = [1,1,1,1]
            can2sparsity[_candidate.reviewerID][select_review] = 0
    

    # can2item = [_CONS.this_asin for _CONS in ITEM_CONSUMER]
    can2item = dict()
    for index, _candidate in enumerate(ITEM_CONSUMER):
        can2item[_candidate.reviewerID] = _candidate.this_asin
        
    if (write_can2item):
        wpickle(opt.pk_fpath, 'can2item', can2item)

    wpickle(opt.pk_fpath, opt.pk_fname, can2sparsity)
    
    pass

def _load(fpath, fname):
    # reload a file to a variable
    with open('{}/review_sparsity_{}.pickle'.format(fpath, fname),  'rb') as file:
        a_dict1 =pickle.load(file)

    print(len(a_dict1))
    pass

if __name__ == "__main__":
    """This process is working for generate sparsity reviews of USER SET."""

    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--mode", 
        default="load", 
        choices=["save", "load"], 
        help="" 
        )
    parser.add_argument('--sqlfile', default='', help="loacl sql cmd file")
    parser.add_argument("--train_test_rand_seed", type=int, default=42, help="rand seed for data seleting")        
    parser.add_argument("--setence_max_len", type=int, default=100, help="Max length of sentence")        
    parser.add_argument('--net_type', default='user_base', help="select net type(user or item base)")
    parser.add_argument('--selectTable', default='clothing_', help="select db table")
    parser.add_argument("--having_interactions", type=int, default=15, help="num of user interactions")        
    parser.add_argument('--num_of_reviews', type=int, default=4, help="number of every user's reviews")
    parser.add_argument('--pk_fpath', default='HANG/data/review_sparsity')
    parser.add_argument('--pk_fname', default='test_3rd')
    parser.add_argument("--select_review", type=int, default=-1, help="")        

    opt = parser.parse_args()

    sen_len = opt.setence_max_len
        
    data_preprocess = Preprocess(
        setence_max_len=sen_len, 
        use_nltk_stopword=False
    )
    if(opt.mode == 'save'):
        _store(
            data_preprocess, 
            opt, 
            testing=True, 
            select_review=opt.select_review,
            write_can2item=not True
            )
    elif(opt.mode == 'load'):
        _load(opt.pk_fpath, opt.pk_fname)        