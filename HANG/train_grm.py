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


    if(opt.mode == "train" or opt.mode == "both"):

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

        STOP = 1
        # Generate `training set batches`
        training_sentences_batches, external_memorys, training_review_rating = data_preprocess.GenerateTrainingBatches(CANDIDATE, candidateObj, voc, 
            net_type = opt.net_type, 
            num_of_reviews=opt.num_of_reviews, 
            batch_size=opt.batchsize,
            get_rating_batch = True)

        review_generation.set_training_batches(training_sentences_batches, external_memorys, candidate_asins, candidate_reviewerIDs, training_batch_labels)

        review_generation.set_label_sentences(label_sen_batch)
        review_generation.set_tune_option(use_pretrain_item_net=True, tuning_item_net=True)
        review_generation.set_decoder_learning_ratio(opt.decoder_learning_ratio)
        review_generation.set_training_review_rating(training_review_rating)

    # Generate testing batches
    if(opt.mode == "eval_mse" or opt.mode == "eval_bleu" or opt.mode == "generation" or opt.mode == "train"):        
        
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
        testing_batches, testing_external_memorys, testing_review_rating = data_preprocess.GenerateTrainingBatches(
            CANDIDATE, 
            candidateObj, 
            voc, 
            net_type = opt.net_type,
            num_of_reviews=opt.num_of_reviews, 
            batch_size=opt.batchsize, 
            testing=True,
            get_rating_batch = True
            )

        review_generation.set_testing_batches(
            testing_batches, 
            testing_external_memorys, 
            testing_batch_labels, 
            testing_asins, 
            testing_reviewerIDs, 
            testing_label_sentences
            )
        
        review_generation.set_object(userObj, itemObj)
        review_generation.set_testing_review_rating(testing_review_rating)
    
    if(opt.use_coverage == 'Y'):
        _use_coverage = True
    else:
        _use_coverage = False

    if(opt.mode == "train" or opt.mode == "both"):
        review_generation.train_grm(
            opt.selectTable, 
            isStoreModel=True, 
            WriteTrainLoss=True, 
            store_every = opt.save_model_freq, 
            use_pretrain_item=False, 
            isCatItemVec=False, 
            concat_rating = True,
            ep_to_store=opt.epoch_to_store,
            pretrain_wordVec=pretrain_wordVec,
            _use_coverage = _use_coverage
            )


    # Testing(chose epoch)
    if(opt.mode == "generation"):

        # Set up asin2title
        review_generation.set_asin2title(
            data_preprocess.load_asin2title(sqlfile='HANG/SQL/cloth_asin2title.sql')
        )
        # Setup epoch being chosen
        chose_epoch = opt.epoch

        # Loading IntraGRU
        IntraGRU = list()
        for idx in range(opt.num_of_reviews):
            model = torch.load(
                R'{}/Model/IntraGRU_idx{}_epoch{}'.format(
                    opt.save_dir, idx, chose_epoch
                    )
                )
            IntraGRU.append(model)
        # Loading InterGRU
        InterGRU = torch.load(R'{}/Model/InterGRU_epoch{}'.format(opt.save_dir, chose_epoch))
        # Loading DecoderModel
        DecoderModel = torch.load(R'{}/Model/DecoderModel_epoch{}'.format(opt.save_dir, chose_epoch))

        # evaluating
        RMSE, batch_bleu_score = review_generation.evaluate_generation(
            IntraGRU, 
            InterGRU, 
            DecoderModel, 
            isCatItemVec=False, 
            concat_rating = True, 
            write_insert_sql=True,
            write_origin=True,
            _use_coverage=_use_coverage,
            _write_mode = 'generate'
            )

        # for num, val in enumerate(batch_bleu_score):
        #     with open('{}/Bleu/blue{}.score_ep{}.txt'.format(opt.save_dir, (num+1), chose_epoch),'a') as file:
        #         file.write('BLEU SCORE {}.ep.{}: {}'.format((num+1), chose_epoch, val))
        #     print('\nBLEU SCORE {}: {}'.format((num+1), val))


    # Testing(chose epoch)
    if(opt.mode == "eval_bleu"):

        # Set up asin2title
        review_generation.set_asin2title(
            data_preprocess.load_asin2title(sqlfile='HANG/SQL/cloth_asin2title.sql')
        )

        for Epoch in range(opt.start_epoch, opt.epoch, opt.save_model_freq):
            # Loading IntraGRU
            IntraGRU = list()
            for idx in range(opt.num_of_reviews):
                model = torch.load(R'{}/Model/IntraGRU_idx{}_epoch{}'.format(opt.save_dir, idx, Epoch))
                IntraGRU.append(model)

            # Loading InterGRU
            InterGRU = torch.load(R'{}/Model/InterGRU_epoch{}'.format(opt.save_dir, Epoch))

            # Loading DecoderModel
            DecoderModel = torch.load(R'{}/Model/DecoderModel_epoch{}'.format(opt.save_dir, Epoch))
        
            # evaluating
            RMSE, _nllloss, batch_bleu_score, average_rouge_score = review_generation.evaluate_generation(
                IntraGRU, 
                InterGRU, 
                DecoderModel, 
                isCatItemVec=False, 
                concat_rating = True,
                write_insert_sql=True,
                write_origin=True,
                _use_coverage=_use_coverage,
                _write_mode = 'evaluate'
                )

            for num, val in enumerate(batch_bleu_score):
                with open('{}/Bleu/blue{}.score.txt'.format(opt.save_dir, (num+1)),'a') as file:
                    file.write('BLEU SCORE {}.ep.{}: {}\n'.format((num+1), Epoch, val))
                print('\nBLEU SCORE {}: {}'.format((num+1), val))

            with open('{}/Bleu/rouge.score.txt'.format(opt.save_dir), 'a') as file:
                file.write('=============================\nEpoch:{}\n'.format(Epoch))
                for _rouge_method, _metrics in average_rouge_score.items():
                    for _key, _val in _metrics.items():
                        file.write('{}. {}: {}\n'.format(_rouge_method, _key, _val))
                        print('{}. {}: {}'.format(_rouge_method, _key, _val))

    # Evaluation
    if(opt.mode == "eval_mse"):

        for Epoch in range(0, opt.epoch, opt.save_model_freq):
            # Loading IntraGRU
            IntraGRU = list()
            for idx in range(opt.num_of_reviews):
                model = torch.load(R'{}/Model/IntraGRU_idx{}_epoch{}'.format(opt.save_dir, idx, Epoch))
                IntraGRU.append(model)

            # Loading InterGRU
            InterGRU = torch.load(R'{}/Model/InterGRU_epoch{}'.format(opt.save_dir, Epoch))        

            # evaluating
            RMSE = review_generation.evaluate_mse(
                IntraGRU, InterGRU, isCatItemVec=True
                )

            print('Epoch:{}\tMSE:{}\t'.format(Epoch, RMSE))

            with open(R'{}/Loss/TestingLoss.txt'.format(opt.save_dir),'a') as file:
                file.write('Epoch:{}\tRMSE:{}\n'.format(Epoch, RMSE))
    pass

def run():
    data_preprocess = Preprocess(setence_max_len=opt.setence_max_len)
    _train_test(data_preprocess)
    pass

if __name__ == "__main__":
    data_preprocess = Preprocess(setence_max_len=opt.setence_max_len)
    _train_test(data_preprocess)

