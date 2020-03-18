from utils import options, visulizeOutput
from utils.preprocessing import Preprocess
from utils.model import IntraReviewGRU, HANN
from utils._ratingregression import RatingRegresstion
from visualization.attention_visualization import Visualization

import datetime
import tqdm
import torch
import torch.nn as nn
from torch import optim
import random

from gensim.models import KeyedVectors
import numpy as np

# Use cuda
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
opt = options.GatherOptions().parse()

# If use pre-train word vector , load .vec
if(opt.use_pretrain_word == 'Y'):
    filename = 'HANG/data/{}festtext_subEmb.vec'.format(opt.selectTable)
    pretrain_words = KeyedVectors.load_word2vec_format(filename, binary=False)


def _train_HANN_model(data_preprocess):

    """
    The model construction base on Item-net model.
    """

    item_net_sql = opt.sqlfile
    base_model_net_type = 'item_base'
    correspond_model_net_type = 'user_base'

    res, itemObj, userObj = data_preprocess.load_data(
        sqlfile=item_net_sql, 
        testing=False, 
        table= opt.selectTable, 
        rand_seed=opt.train_test_rand_seed
        )  # for clothing.

    # Generate voc & (User or Item) information , CANDIDATE could be USER or ITEM
    voc, ITEM, candiate2index = data_preprocess.generate_candidate_voc(
        res, 
        having_interaction=opt.having_interactions, 
        net_type = base_model_net_type
        )

    # pre-train words
    if(opt.use_pretrain_word == 'Y'):
        weights_matrix = data_preprocess.load_pretain_word(voc, pretrain_words)
        weights_tensor = torch.FloatTensor(weights_matrix)
        pretrain_wordVec = nn.Embedding.from_pretrained(weights_tensor).to(device)           
    else:
        pretrain_wordVec = None
    
    # Generate train set && candidate
    training_batch_labels, candidate_asins, candidate_reviewerIDs, _ = data_preprocess.get_train_set(ITEM, 
        itemObj, 
        userObj, 
        voc,
        batchsize=opt.batchsize, 
        num_of_reviews=5, 
        num_of_rating=1
        )

    if(base_model_net_type == 'user_base'):
        candidateObj = itemObj
    elif(base_model_net_type == 'item_base'):
        candidateObj = userObj

    # Generate `training set batches`
    # training_batches contain {sentences batch, rating batch, length batch}
    training_batches, training_asin_batches = data_preprocess.GenerateTrainingBatches(
        ITEM, 
        candidateObj, 
        voc, 
        net_type = base_model_net_type, 
        num_of_reviews=opt.num_of_reviews, 
        batch_size=opt.batchsize
        )
    

    """
    Generate user-net training data.
    """
    num_of_reviews_unet = 4

    user_base_sql = R'HANG/SQL/cloth_candidate_asin.sql'
    res, itemObj, userObj = data_preprocess.load_data(
        sqlfile=user_base_sql, 
        testing=False, 
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

    # Generate correspond net batches
    correspond_batches, correspond_asin_batches = data_preprocess.GenerateTrainingBatches(
        ITEM_CONSUMER, itemObj, voc, 
        net_type = correspond_model_net_type,
        num_of_reviews= num_of_reviews_unet, 
        batch_size=opt.batchsize,
        testing=True
        )

    # Start to train model by `rating regression`
    rating_regresstion = RatingRegresstion(
        device, opt.net_type, opt.save_dir, voc, data_preprocess, 
        training_epoch=opt.epoch, latent_k=opt.latentK, 
        batch_size=opt.batchsize,
        hidden_size=opt.hidden, clip=opt.clip,
        num_of_reviews = opt.num_of_reviews, 
        intra_method=opt.intra_attn_method , inter_method=opt.inter_attn_method,
        learning_rate=opt.lr, dropout=opt.dropout
        )

    """Setting training setup"""
    rating_regresstion.set_training_batches(training_batches, training_asin_batches, candidate_asins, candidate_reviewerIDs, training_batch_labels)
    rating_regresstion.set_correspond_batches(correspond_batches)           # this method for hybird only
    rating_regresstion.set_correspond_num_of_reviews(num_of_reviews_unet)   # this method for hybird only
    rating_regresstion.set_correspond_external_memorys(correspond_asin_batches)    # this method for hybird only

    """Creating testing batches"""
    # Loading testing data from database
    res, itemObj, userObj = data_preprocess.load_data(
        sqlfile=opt.sqlfile, 
        testing=True, 
        table=opt.selectTable, 
        rand_seed=opt.train_test_rand_seed
        )   # clothing

    # If mode = test, won't generate a new voc.
    CANDIDATE, candiate2index = data_preprocess.generate_candidate_voc(
        res, having_interaction=opt.having_interactions, generate_voc=False, 
        net_type = opt.net_type
        )

    testing_batch_labels, testing_asins, testing_reviewerIDs, _ = data_preprocess.get_train_set(CANDIDATE, 
        itemObj, 
        userObj, 
        voc,
        batchsize=opt.batchsize, 
        num_of_reviews=5, 
        num_of_rating=1
        )

    # Generate testing batches
    testing_batches, testing_external_memorys = data_preprocess.GenerateTrainingBatches(
        CANDIDATE, userObj, voc, net_type = opt.net_type,
        num_of_reviews=opt.num_of_reviews, batch_size=opt.batchsize, 
        testing=True
        )



    if(opt.hybird == 'Y'):

        user_base_sql = R'HANG/SQL/cloth_candidate_asin.sql'
        res, itemObj, userObj = data_preprocess.load_data(
            sqlfile = user_base_sql, 
            testing = True, 
            table = opt.selectTable, 
            rand_seed = opt.train_test_rand_seed
            )  # for clothing.

        # Generate voc & (User or Item) information , CANDIDATE could be USER or ITEM
        USER, uid2index = data_preprocess.generate_candidate_voc(
            res, 
            having_interaction = opt.having_interactions, 
            net_type = 'user_base',
            generate_voc = False
            )

        # Create item consumer list
        ITEM_CONSUMER = list()
        for _item in CANDIDATE:
            candidate_uid = _item.this_reviewerID[5]
            # u_index = userObj.reviewerID2index[candidate_uid]
            user_index = uid2index[candidate_uid]

            ITEM_CONSUMER.append(USER[user_index])

        """Enable useing sparsity review"""
        if(opt.use_sparsity_review == 'Y'):

            # loading sparsity review
            can2sparsity = data_preprocess.load_sparsity_reviews(
                'HANG/data/review_sparsity', 
                'test'
                )
            # setup can2sparsity
            rating_regresstion.set_can2sparsity(can2sparsity)

            # Replace reviews by sparsity
            for _index, user in enumerate(ITEM_CONSUMER):
                # load target user's sparsity list
                sparsity_list = can2sparsity[user.reviewerID]
                
                # Replace reviews to null by `sparsity_list`
                for _num_of_review , _val in enumerate(sparsity_list):
                    if(_val == 0):
                        ITEM_CONSUMER[_index].sentences[_num_of_review] = ''
            
        # Generate `training correspond set batches`
        correspond_batches, correspond_asin_batches = data_preprocess.GenerateTrainingBatches(
            ITEM_CONSUMER, 
            itemObj, 
            voc, 
            net_type = 'user_base', 
            num_of_reviews= 4, 
            batch_size=opt.batchsize,
            testing=True
            )
        
        rating_regresstion.set_testing_correspond_batches(correspond_batches)

        pass

    """Setting testing setup"""
    rating_regresstion.set_testing_batches(
        testing_batches, 
        testing_external_memorys,
        testing_batch_labels, 
        testing_asins, 
        testing_reviewerIDs
    )

    """Start training"""
    if(opt.mode == 'both' or opt.mode == 'train'):
        rating_regresstion.hybird_train(
            opt.selectTable, 
            isStoreModel=True, 
            WriteTrainLoss=True, 
            store_every = opt.save_model_freq, 
            use_pretrain_item=False, 
            isCatItemVec=not True, 
            pretrain_wordVec=pretrain_wordVec
            )


    """Start testing"""
    if(opt.mode == "test"):

        for Epoch in range(0, opt.epoch, opt.save_model_freq):
            # Loading IntraGRU
            IntraGRU = list()
            for idx in range(opt.num_of_reviews):
                model = torch.load(R'{}/Model/IntraGRU_idx{}_epoch{}'.format(opt.save_dir, idx, Epoch))
                IntraGRU.append(model)
            
            # Loading correspond IntraGRU
            correspond_IntraGRU = list()
            for idx in range(opt.num_of_correspond_reviews):
                model = torch.load(R'{}/Model/correspond_IntraGRU_idx{}_epoch{}'.format(opt.save_dir, idx, Epoch))
                correspond_IntraGRU.append(model)
            
            # Loading InterGRU
            InterGRU = torch.load(R'{}/Model/InterGRU_epoch{}'.format(opt.save_dir, Epoch))
            correspond_InterGRU = torch.load(R'{}/Model/correspond_InterGRU_epoch{}'.format(opt.save_dir, Epoch))
            
            # Loading MFC
            MFC = torch.load(R'{}/Model/MFC_epoch{}'.format(opt.save_dir, Epoch))

            # Evaluating hybird model
            RMSE = rating_regresstion._hybird_evaluate(
                IntraGRU, InterGRU, correspond_IntraGRU, correspond_InterGRU, MFC,
                testing_batches, testing_external_memorys, testing_batch_labels, testing_asins, testing_reviewerIDs,
                correspond_batches, 
                isCatItemVec=not True, visulize_attn_epoch=opt.epoch
                )

            print('Epoch:{}\tMSE:{}\t'.format(Epoch, RMSE))

            if(opt.minor_path==''):
                with open(R'{}/Loss/TestingLoss.txt'.format(opt.save_dir),'a') as file:
                    file.write('Epoch:{}\tRMSE:{}\n'.format(Epoch, RMSE))
            else:
                with open(R'{}/Loss/{}'.format(opt.save_dir, opt.minor_path),'a') as file:
                    file.write('Epoch:{}\tRMSE:{}\n'.format(Epoch, RMSE))

if __name__ == "__main__":

    data_preprocess = Preprocess(setence_max_len=opt.setence_max_len, use_nltk_stopword=opt.use_nltk_stopword)
    _train_HANN_model(data_preprocess)

