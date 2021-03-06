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
import time

# Use cuda
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
opt = options.GatherOptions().parse()

# If use pre-train word vector , load .vec
if(opt.use_pretrain_word == 'Y'):
    print('\nLoading pre-train word vector...') 
    st = time.time()    
    filename = 'HANG/data/{}festtext_subEmb.vec'.format(opt.selectTable)
    pretrain_words = KeyedVectors.load_word2vec_format(filename, binary=False)
    print('Loading complete. [{}]'.format(time.time()-st))


def _train_test(data_preprocess):

    res, itemObj, userObj = data_preprocess.load_data(
        sqlfile=opt.sqlfile, 
        mode='train', 
        table= opt.selectTable, 
        rand_seed=opt.train_test_rand_seed
        )  # for clothing.

    # Generate voc & (User or Item) information , CANDIDATE could be USER or ITEM
    voc, CANDIDATE, candiate2index = data_preprocess.generate_candidate_voc(
        res, 
        having_interaction=opt.having_interactions, 
        net_type = opt.net_type
        )

    # pre-train words
    if(opt.use_pretrain_word == 'Y'):
        weights_matrix = data_preprocess.load_pretain_word(voc, pretrain_words)
        weights_tensor = torch.FloatTensor(weights_matrix)
        pretrain_wordVec = nn.Embedding.from_pretrained(weights_tensor).to(device)           
    else:
        pretrain_wordVec = None


    """
    Construct RGM task
    """
    review_generation = ReviewGeneration(device, opt.net_type, opt.save_dir, voc, data_preprocess, 
        training_epoch=opt.epoch, latent_k=opt.latentK, batch_size=opt.batchsize, hidden_size=opt.hidden, clip=opt.clip,
        num_of_reviews = opt.num_of_reviews, 
        intra_method=opt.intra_attn_method , inter_method=opt.inter_attn_method,
        learning_rate=opt.lr, dropout=opt.dropout)        


    if(opt.mode == "train" or opt.mode == "generation"):

        # Generate train set && candidate
        training_batch_labels, candidate_asins, candidate_reviewerIDs, label_sen_batch = data_preprocess.get_train_set(CANDIDATE, 
            itemObj, 
            userObj, 
            voc,
            batchsize=opt.batchsize, 
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

        review_generation.set_training_batches(training_sentences_batches, external_memorys, candidate_asins, candidate_reviewerIDs, training_batch_labels)

        review_generation.set_label_sentences(label_sen_batch)
        review_generation.set_decoder_learning_ratio(opt.decoder_learning_ratio)

    # Generate testing batches
    if(opt.mode == "eval_mse" or opt.mode == "generation" or opt.mode == "train"):
        
        review_generation.set_testing_set(
            test_on_train_data = opt.test_on_traindata
            )

        # Loading testing data from database
        res, itemObj, userObj = data_preprocess.load_data(
            sqlfile=opt.sqlfile, 
            mode='validation', 
            table=opt.selectTable, 
            rand_seed=opt.train_test_rand_seed, 
            test_on_train_data=review_generation.test_on_train_data
            )  
        
        # If mode:`test` , won't generate a new voc.
        CANDIDATE, candiate2index = data_preprocess.generate_candidate_voc(res, having_interaction=opt.having_interactions, generate_voc=False, 
            net_type = opt.net_type)

        testing_batch_labels, testing_asins, testing_reviewerIDs, testing_label_sentences = data_preprocess.get_train_set(
            CANDIDATE, 
            itemObj, 
            userObj, 
            voc,
            batchsize=opt.batchsize, 
            num_of_reviews=5, 
            num_of_rating=1,
            net_type=opt.net_type,
            testing=True,
            mode='generate'            
            )

        if(opt.net_type == 'user_base'):
            candidateObj = itemObj
        elif(opt.net_type == 'item_base'):
            candidateObj = userObj

        # Generate testing batches
        testing_batches, testing_external_memorys = data_preprocess.GenerateTrainingBatches(
            CANDIDATE, 
            candidateObj, 
            voc, 
            net_type = opt.net_type,
            num_of_reviews=opt.num_of_reviews, 
            batch_size=opt.batchsize, 
            testing=True
            )

        review_generation.set_object(userObj, itemObj)
        review_generation.set_testing_batches(
            testing_batches, 
            testing_external_memorys, 
            testing_batch_labels, 
            testing_asins, 
            testing_reviewerIDs, 
            testing_label_sentences
            )
    
    if(opt.use_coverage == 'Y'):
        _use_coverage = True
    else:
        _use_coverage = False

    if(opt.mode == "train"):
        review_generation.train_nrt(
            opt.selectTable, 
            isStoreModel=True, 
            WriteTrainLoss=True,
            store_every = opt.save_model_freq, 
            use_pretrain_item=False, 
            ep_to_store=opt.epoch_to_store,
            pretrain_wordVec=pretrain_wordVec,
            voc=voc
            )

    if(opt.mode == "test"):

        NRT = torch.load(R'{}/Model/NRT_epoch{}'.format(opt.save_dir, opt.epoch))
        NRT.eval()

        NRTD = torch.load(R'{}/Model/NRTD_epoch{}'.format(opt.save_dir, opt.epoch))
        NRTD.eval()

        _ = review_generation.evaluate_nrt_generation(
            NRT, 
            NRTD
            )


    if(opt.mode == "generation"):

        NRT = torch.load(R'{}/Model/NRT_epoch{}'.format(opt.save_dir, opt.epoch))
        NRT.eval()

        NRTD = torch.load(R'{}/Model/NRTD_epoch{}'.format(opt.save_dir, opt.epoch))
        NRTD.eval()

        _ = review_generation.evaluate_nrt_generation(
            NRT, 
            NRTD
            )

    pass

if __name__ == "__main__":
    data_preprocess = Preprocess(setence_max_len=opt.setence_max_len)
    _train_test(data_preprocess)

