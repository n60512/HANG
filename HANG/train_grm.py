from utils import options, visulizeOutput
from utils.preprocessing import Preprocess
from utils.model import IntraReviewGRU, HANN

from utils._reviewgeneration import ReviewGeneration
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


def _single_model(data_preprocess):

    res, itemObj, userObj = data_preprocess.load_data(sqlfile=opt.sqlfile, testing=False, table= opt.selectTable, rand_seed=opt.train_test_rand_seed)  # for clothing.

    # Generate voc & (User or Item) information , CANDIDATE could be USER or ITEM
    voc, CANDIDATE, candiate2index = data_preprocess.generate_candidate_voc(res, 
        having_interaction=opt.having_interactions, 
        net_type = opt.net_type)

    # pre-train words
    if(opt.use_pretrain_word == 'Y'):
        weights_matrix = data_preprocess.load_pretain_word(voc, pretrain_words)
        weights_tensor = torch.FloatTensor(weights_matrix)
        pretrain_wordVec = nn.Embedding.from_pretrained(weights_tensor).to(device)           
    else:
        pretrain_wordVec = None


    if(opt.mode == "train" or opt.mode == "both" or opt.mode == "generation"):

        # Generate train set && candidate
        training_batch_labels, candidate_asins, candidate_reviewerIDs, label_sen_batch = data_preprocess.get_train_set(CANDIDATE, 
            itemObj, 
            userObj, 
            voc,
            batchsize=40, 
            num_of_reviews=5, 
            num_of_rating=1,
            net_type=opt.net_type,
            mode='generate'
            )

        if(opt.net_type == 'user_base'):
            candidateObj = itemObj
        elif(opt.net_type == 'item_base'):
            candidateObj = userObj

        # Generate `training set batches`
        training_sentences_batches, external_memorys = data_preprocess.GenerateTrainingBatches(CANDIDATE, candidateObj, voc, 
            net_type = opt.net_type, 
            num_of_reviews=opt.num_of_reviews, 
            batch_size=opt.batchsize)

    if(True):

        review_generation = ReviewGeneration(device, opt.net_type, opt.save_dir, voc, data_preprocess, 
            training_epoch=opt.epoch, latent_k=opt.latentK, hidden_size=opt.hidden, clip=opt.clip,
            num_of_reviews = opt.num_of_reviews, 
            intra_method=opt.intra_attn_method , inter_method=opt.inter_attn_method,
            learning_rate=opt.lr, dropout=opt.dropout)

        review_generation.set_training_batches(training_sentences_batches, external_memorys, candidate_asins, candidate_reviewerIDs, training_batch_labels)

        review_generation.set_label_sentences(label_sen_batch)
        review_generation.set_tune_option(user_pretrain_hann=False, tuning_hann=True)
        review_generation.set_decoder_learning_ratio(opt.decoder_learning_ratio)

        review_generation.train_grm(opt.selectTable, isStoreModel=True, WriteTrainLoss=True, store_every = opt.save_model_freq, 
            use_pretrain_item=False, isCatItemVec=True, pretrain_wordVec=pretrain_wordVec)




    stop = 1

    # Generate testing batches
    if(opt.mode == "test" or opt.mode == "showAttn" or opt.mode == "both"):

        # Loading testing data from database
        res, itemObj, userObj = data_preprocess.load_data(sqlfile=opt.sqlfile, testing=True, table=opt.selectTable, rand_seed=opt.train_test_rand_seed)   # clothing
        # If mode = test, won't generate a new voc.
        CANDIDATE, candiate2index = data_preprocess.generate_candidate_voc(res, having_interaction=opt.having_interactions, generate_voc=False, 
            net_type = opt.net_type)

        testing_batch_labels, candidate_asins, candidate_reviewerIDs, label_sen_batch = data_preprocess.get_train_set(CANDIDATE, 
            itemObj, 
            userObj, 
            voc,
            batchsize=40, 
            num_of_reviews=5, 
            num_of_rating=1,
            net_type=opt.net_type,
            mode='generate'            
            )

        # Generate testing batches
        testing_batches, testing_asin_batches = data_preprocess.GenerateTrainingBatches(CANDIDATE, userObj, voc, net_type = opt.net_type,
            num_of_reviews=opt.num_of_reviews, batch_size=opt.batchsize, testing=True)


    # Testing
    if(opt.mode == "test" or opt.mode == "both"):
        # Evaluation (testing data)
        for Epoch in range(0, opt.epoch, opt.save_model_freq):
            # Loading IntraGRU
            IntraGRU = list()
            for idx in range(opt.num_of_reviews):
                model = torch.load(R'{}/Model/IntraGRU_idx{}_epoch{}'.format(opt.save_dir, idx, Epoch))
                IntraGRU.append(model)

            # Loading InterGRU
            InterGRU = torch.load(R'{}/Model/InterGRU_epoch{}'.format(opt.save_dir, Epoch))

            rating_regresstion = RatingRegresstion(device, opt.net_type, opt.save_dir, voc, data_preprocess, 
                training_epoch=opt.epoch, latent_k=opt.latentK, hidden_size=opt.hidden, clip=opt.clip,
                num_of_reviews = opt.num_of_reviews, 
                intra_method=opt.intra_attn_method , inter_method=opt.inter_attn_method,
                learning_rate=opt.lr, dropout=opt.dropout)            
                
            RMSE = rating_regresstion.evaluate(IntraGRU, InterGRU, testing_batches, testing_asin_batches, testing_batch_labels, candidate_asins, candidate_reviewerIDs, 
                isCatItemVec=True, visulize_attn_epoch=opt.epoch)

            print('Epoch:{}\tMSE:{}\t'.format(Epoch, RMSE))

            with open(R'{}/Loss/TestingLoss.txt'.format(opt.save_dir),'a') as file:
                file.write('Epoch:{}\tRMSE:{}\n'.format(Epoch, RMSE))    

    # Testing (with showing attention weight)
    if(opt.mode == "showAttn"):
        # Loading IntraGRU
        IntraGRU = list()
        for idx in range(opt.num_of_reviews):
            model = torch.load(R'{}/Model/IntraGRU_idx{}_epoch{}'.format(opt.save_dir, idx, opt.visulize_attn_epoch))
            IntraGRU.append(model)

        # Loading InterGRU
        InterGRU = torch.load(R'{}/Model/InterGRU_epoch{}'.format(opt.save_dir, opt.visulize_attn_epoch))

        # evaluating
        RMSE = evaluate(IntraGRU, InterGRU, testing_batches, testing_asin_batches, testing_batch_labels, candidate_asins, candidate_reviewerIDs, 
            isCatItemVec=True, isWriteAttn=True, userObj=userObj)

    pass


if __name__ == "__main__":


    data_preprocess = Preprocess(setence_max_len=opt.setence_max_len, use_nltk_stopword=opt.use_nltk_stopword)
    
    _single_model(data_preprocess)

