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
import matplotlib.pyplot as plt

# Use cuda
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
opt = options.GatherOptions().parse()

"""Loading pretrain fasttext embedding"""
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
        mode='train', 
        table= opt.selectTable, 
        rand_seed=opt.train_test_rand_seed
        )

    # Generate voc & (User or Item) information , CANDIDATE could be USER or ITEM
    voc, ITEM, candiate2index = data_preprocess.generate_candidate_voc(
        res, 
        having_interaction=opt.having_interactions, 
        net_type = base_model_net_type
        )

    # a = voc.word2index['<number>']
    # b = voc.word2count['<number>']

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
    training_batches, training_asin_batches, training_review_rating = data_preprocess.GenerateTrainingBatches(
        ITEM, 
        candidateObj, 
        voc, 
        net_type = base_model_net_type, 
        num_of_reviews=opt.num_of_reviews, 
        batch_size=opt.batchsize,
        get_rating_batch = True
        )
    
    """
    Generate user-net training data.
    """
    num_of_reviews_unet = opt.num_of_correspond_reviews

    # user_base_sql = R'HANG/SQL/cloth_candidate_asin.sql'
    # user_base_sql = R'HANG/SQL/cloth_candidate_test_on_trained.sql' #0630 test on trained
    user_base_sql = R'HANG/SQL/_all_interaction6_item.candidate.user.sql'
    # user_base_sql = R'HANG/SQL/cloth_candidate_asin_without_rm_sw.sql'    # original
    # user_base_sql = 'HANG/SQL/cloth_interaction@6_userbase.sample.sql'
    res, itemObj, userObj = data_preprocess.load_data(
        sqlfile=user_base_sql, 
        mode='train', 
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
    correspond_batches, correspond_asin_batches, correspond_review_rating = data_preprocess.GenerateTrainingBatches(
        ITEM_CONSUMER, itemObj, voc, 
        net_type = correspond_model_net_type,
        num_of_reviews= num_of_reviews_unet, 
        batch_size=opt.batchsize,
        testing=True,
        get_rating_batch = True
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
    rating_regresstion.set_correspond_batches(correspond_batches)                   # this method for hybird only
    rating_regresstion.set_correspond_num_of_reviews(num_of_reviews_unet)           # this method for hybird only
    rating_regresstion.set_correspond_external_memorys(correspond_asin_batches)     # this method for hybird only
    rating_regresstion.set_training_review_rating(training_review_rating, correspond_review_rating)

    rating_regresstion.set_candidate_object(userObj, itemObj)

    # Chose dataset for (train / validation / test)
    if (opt.mode == 'train'):
        _sql_mode = 'validation'
        pass
    elif (opt.mode == 'test' or opt.mode == 'attention'):
        _sql_mode = 'test'
        _sql_mode = 'validation'
        pass

    """Creating testing batches"""
    # Loading testing data from database
    res, itemObj, userObj = data_preprocess.load_data(
        sqlfile=opt.sqlfile, 
        mode=_sql_mode, 
        table=opt.selectTable, 
        rand_seed=opt.train_test_rand_seed
        )

    # If mode = test, won't generate a new voc.
    CANDIDATE, candiate2index = data_preprocess.generate_candidate_voc(
        res, having_interaction=opt.having_interactions, generate_voc=False, 
        net_type = opt.net_type
        )

    testing_batch_labels, testing_asins, testing_reviewerIDs, _ = data_preprocess.get_train_set(
        CANDIDATE, 
        itemObj, 
        userObj, 
        voc,
        batchsize=opt.batchsize, 
        num_of_reviews=5, 
        num_of_rating=1
        )

    # Generate testing batches
    testing_batches, testing_external_memorys, testing_review_rating = data_preprocess.GenerateTrainingBatches(
        CANDIDATE, userObj, voc, net_type = opt.net_type,
        num_of_reviews=opt.num_of_reviews, batch_size=opt.batchsize, 
        testing=True,
        get_rating_batch = True
        )

    if(opt.hybird == 'Y'):

        """Select testing set (`normal` or `GENERATIVE`)"""
        if(opt.sqlfile_fill_user==''):
            # user_base_sql = R'HANG/SQL/cloth_candidate_asin.sql'
            
            # user_base_sql = R'HANG/SQL/cloth_candidate_test_on_trained.sql' #0630 test on trained
            user_base_sql = R'HANG/SQL/_all_interaction6_item.candidate.user.sql'
            # user_base_sql = R'HANG/SQL/cloth_candidate_asin_without_rm_sw.sql'    # original
            # user_base_sql = 'HANG/SQL/cloth_interaction@6_userbase.sample.sql'
        else:
            user_base_sql = opt.sqlfile_fill_user   # select the generative table

        res, itemObj, userObj = data_preprocess.load_data(
            sqlfile = user_base_sql, 
            mode=_sql_mode, 
            table = opt.selectTable, 
            rand_seed = opt.train_test_rand_seed,
            num_of_generative=opt.num_of_generative
            )  

        # Generate USER information 
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
            user_index = uid2index[candidate_uid]

            ITEM_CONSUMER.append(USER[user_index])

        """Enable useing sparsity review (training set `OFF`)"""
        """Using this when testing on sparsity reviews"""
        if(opt.use_sparsity_review == 'Y'):
            # loading sparsity review
            can2sparsity = data_preprocess.load_sparsity_reviews(
                opt.sparsity_pickle
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
        correspond_batches, correspond_asin_batches, correspond_review_rating = data_preprocess.GenerateTrainingBatches(
            ITEM_CONSUMER, 
            itemObj, 
            voc, 
            net_type = 'user_base', 
            num_of_reviews= opt.num_of_correspond_reviews, 
            batch_size=opt.batchsize,
            testing=True,
            get_rating_batch = True
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

    rating_regresstion.set_testing_review_rating(
        testing_review_rating, 
        correspond_review_rating
        )

    # Set random sparsity
    if(opt.concat_item == 'Y'):
        concat_item = True
    else:
        concat_item = False


    # Concat. rating embedding
    if(opt.concat_review_rating == 'Y'):
        concat_rating = True
    else:
        concat_rating = False

    # Set random sparsity
    if(opt._ran_sparsity == 'Y'):
        _ran_sparsity = True
        _reviews_be_chosen = opt._reviews_be_chosen
    else:
        _ran_sparsity = False
        _reviews_be_chosen = None
    
    rating_regresstion.set_ran_sparsity(
        _ran_sparsity=_ran_sparsity, 
        _reviews_be_chosen=_reviews_be_chosen
        )

    """Start training"""
    if(opt.mode == 'train'):
        rating_regresstion.hybird_train(
            opt.selectTable, 
            isStoreModel=True, 
            WriteTrainLoss=True, 
            store_every = opt.save_model_freq, 
            use_pretrain_item=False, 
            isCatItemVec=concat_item, 
            concat_rating=concat_rating,
            pretrain_wordVec=pretrain_wordVec,
            epoch_to_store = opt.epoch_to_store
            )


    """Start testing"""
    if(opt.mode == "test"):

        for Epoch in range(opt.start_epoch, opt.epoch, opt.save_model_freq):
            # Loading IntraGRU
            IntraGRU = list()
            for idx in range(opt.num_of_reviews):
                model = torch.load(R'{}/Model/IntraGRU_idx{}_epoch{}'.format(opt.save_dir, idx, Epoch))
                model.eval()
                IntraGRU.append(model)
            
            # Loading correspond IntraGRU
            correspond_IntraGRU = list()
            for idx in range(opt.num_of_correspond_reviews):
                model = torch.load(R'{}/Model/correspond_IntraGRU_idx{}_epoch{}'.format(opt.save_dir, idx, Epoch))
                model.eval()
                correspond_IntraGRU.append(model)
            
            # Loading InterGRU
            InterGRU = torch.load(R'{}/Model/InterGRU_epoch{}'.format(opt.save_dir, Epoch))
            InterGRU.eval()
            correspond_InterGRU = torch.load(R'{}/Model/correspond_InterGRU_epoch{}'.format(opt.save_dir, Epoch))
            correspond_InterGRU.eval()

            # Loading MFC
            MFC = torch.load(R'{}/Model/MFC_epoch{}'.format(opt.save_dir, Epoch))
            MFC.eval()

            # Evaluating hybird model
            RMSE, Accuracy, cnf_matrix = rating_regresstion._hybird_evaluate(
                IntraGRU, InterGRU, correspond_IntraGRU, correspond_InterGRU, MFC,
                testing_batches, testing_external_memorys, testing_batch_labels, testing_asins, testing_reviewerIDs,
                correspond_batches, 
                isCatItemVec=concat_item, 
                concat_rating= concat_rating,
                visulize_attn_epoch=opt.epoch
                )

            # Write confusion matrix
            plt.figure()
            rating_regresstion.plot_confusion_matrix(
                cnf_matrix, 
                classes = ['1pt', '2pt', '3pt', '4pt', '5pt'],
                normalize = True,
                title = 'confusion matrix'
                )

            plt.savefig('{}/Loss/Confusion.Matrix/_{}.png'.format(
                opt.save_dir,
                Epoch
            ))

            print('Epoch:{}\tMSE:{}\tAccuracy:{}'.format(Epoch, RMSE, Accuracy))

            if(opt.minor_path==''):
                with open(R'{}/Loss/TestingLoss.txt'.format(opt.save_dir),'a') as file:
                    file.write('Epoch:{}\tRMSE:{}\n'.format(Epoch, RMSE))
                with open(R'{}/Loss/Accuracy.txt'.format(opt.save_dir),'a') as file:
                    file.write('Epoch:{}\tAccuracy:{}\n'.format(Epoch, Accuracy))                      
            else:
                with open(R'{}/Loss/{}'.format(opt.save_dir, opt.minor_path),'a') as file:
                    file.write('Epoch:{}\tRMSE:{}\n'.format(Epoch, RMSE))
    
    if(opt.mode == "attention"):

        # Loading IntraGRU
        IntraGRU = list()
        for idx in range(opt.num_of_reviews):
            model = torch.load(R'{}/Model/IntraGRU_idx{}_epoch{}'.format(opt.save_dir, idx, opt.visulize_attn_epoch))
            IntraGRU.append(model)

        # Loading correspond IntraGRU
        correspond_IntraGRU = list()
        for idx in range(opt.num_of_correspond_reviews):
            model = torch.load(R'{}/Model/correspond_IntraGRU_idx{}_epoch{}'.format(opt.save_dir, idx, opt.visulize_attn_epoch))
            correspond_IntraGRU.append(model)

        # Loading InterGRU
        InterGRU = torch.load(R'{}/Model/InterGRU_epoch{}'.format(opt.save_dir, opt.visulize_attn_epoch))
        correspond_InterGRU = torch.load(R'{}/Model/correspond_InterGRU_epoch{}'.format(opt.save_dir, opt.visulize_attn_epoch))
        
        # Loading MFC
        MFC = torch.load(R'{}/Model/MFC_epoch{}'.format(opt.save_dir, opt.visulize_attn_epoch))


        # Evaluating hybird model
        RMSE, Accuracy, cnf_matrix = rating_regresstion._hybird_evaluate(
            IntraGRU, InterGRU, correspond_IntraGRU, correspond_InterGRU, MFC,
            testing_batches, testing_external_memorys, testing_batch_labels, testing_asins, testing_reviewerIDs,
            correspond_batches, 
            isCatItemVec=concat_item, 
            concat_rating= concat_rating,
            visulize_attn_epoch=opt.visulize_attn_epoch,
            isWriteAttn=True
            )

        print('Epoch:{}\tMSE:{}\t'.format(opt.visulize_attn_epoch, RMSE))

if __name__ == "__main__":

    data_preprocess = Preprocess(setence_max_len=opt.setence_max_len)
    _train_HANN_model(data_preprocess)

