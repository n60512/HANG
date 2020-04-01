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


def _single_model(data_preprocess):

    res, itemObj, userObj = data_preprocess.load_data(sqlfile=opt.sqlfile, testing=False, table= opt.selectTable, rand_seed=opt.train_test_rand_seed)  # for clothing.

    # Generate voc & (User or Item) information , CANDIDATE could be USER or ITEM
    voc, CANDIDATE = data_preprocess.generate_candidate_voc(res, 
        having_interaction=opt.having_interactions, 
        net_type = opt.net_type)

    # pre-train words
    if(opt.use_pretrain_word == 'Y'):
        weights_matrix = data_preprocess.load_pretain_word(voc, pretrain_words)
        weights_tensor = torch.FloatTensor(weights_matrix)
        pretrain_wordVec = nn.Embedding.from_pretrained(weights_tensor).to(device)           
    else:
        pretrain_wordVec = None


    if(opt.mode == "train" or opt.mode == "both"):

        # Generate train set && candidate
        training_batch_labels, candidate_asins, candidate_reviewerIDs = data_preprocess.get_train_set(CANDIDATE, 
            itemObj, 
            userObj, 
            voc,
            batchsize=40, 
            num_of_reviews=5, 
            num_of_rating=1
            )

        if(opt.net_type == 'user_base'):
            candidateObj = itemObj
        elif(opt.net_type == 'item_base'):
            candidateObj = userObj

        # Generate `training set batches`
        training_batches, training_asin_batches = data_preprocess.GenerateTrainingBatches(CANDIDATE, candidateObj, voc, 
            net_type = opt.net_type, 
            num_of_reviews=opt.num_of_reviews, 
            batch_size=opt.batchsize)

    if(True):
        # Train model by `rating regression`
        rating_regresstion = RatingRegresstion(device, opt.net_type, opt.save_dir, voc, data_preprocess, 
            training_epoch=opt.epoch, latent_k=opt.latentK, hidden_size=opt.hidden, clip=opt.clip,
            num_of_reviews = opt.num_of_reviews, 
            intra_method=opt.intra_attn_method , inter_method=opt.inter_attn_method,
            learning_rate=opt.lr, dropout=opt.dropout)

        rating_regresstion.set_training_batchse(training_batches, training_asin_batches, candidate_asins, candidate_reviewerIDs, training_batch_labels)
        rating_regresstion.train(opt.selectTable, isStoreModel=True, WriteTrainLoss=True, store_every = opt.save_model_freq, 
            use_pretrain_item=False, isCatItemVec=True, pretrain_wordVec=pretrain_wordVec)


    # Generate testing batches
    if(opt.mode == "test" or opt.mode == "showAttn" or opt.mode == "both"):

        # Loading testing data from database
        res, itemObj, userObj = data_preprocess.load_data(sqlfile=opt.sqlfile, testing=True, table=opt.selectTable, rand_seed=opt.train_test_rand_seed)   # clothing
        # If mode = test, won't generate a new voc.
        CANDIDATE = data_preprocess.generate_candidate_voc(res, having_interaction=opt.having_interactions, generate_voc=False, 
            net_type = opt.net_type)

        testing_batch_labels, candidate_asins, candidate_reviewerIDs = data_preprocess.get_train_set(CANDIDATE, 
            itemObj, 
            userObj, 
            voc,
            batchsize=40, 
            num_of_reviews=5, 
            num_of_rating=1
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
    
    if(opt.net_type == 'user_base' or opt.net_type == 'item_base'):
        _single_model(data_preprocess)
    elif(opt.net_type == 'hybird'):
        pass



# def randomSelectNoneReview(tensor_, itemEmbedding, type_='z', randomSetup =-1):
#     # tensor_ size:[seq_size, batch_size, hidden_size]

#     # Iterate each user(batch) to give 'Random Val.'
#     for user in range(tensor_.size()[1]):

#         if(randomSetup==-1):
#             # Random amount of reviews (max: all reviews)
#             random_amount_of_reviews = random.randint(0, tensor_.size()[0]-1)
#         else:
#             random_amount_of_reviews = randomSetup

#         for i_ in range(random_amount_of_reviews):
#             select_random_seq = random.randint(0, tensor_.size()[0]-1)    
#             # Give random hidden a NONE REVIEW
#             if(type_=='z'):
#                 tensor_[select_random_seq][user] = torch.zeros(tensor_.size()[2], dtype=torch.float)
#             elif(type_=='r'):
#                 pass
#             elif(type_=='v'):
#                 tensor_[select_random_seq][user] = itemEmbedding[select_random_seq][user]
#     return tensor_





# # Training each iteraction
# def trainIteration(IntraGRU, InterGRU, IntraGRU_optimizer, InterGRU_optimizer, training_batches, training_item_batches,
#     candidate_items, candidate_users, training_batch_labels, isCatItemVec=False):
    
#     # Initialize this epoch loss
#     epoch_loss = 0

#     for batch_ctr in tqdm.tqdm(range(len(training_batches[0]))): # amount of batches
#         # Run multiple label for training 
#         for idx in range(len(training_batch_labels)):

#             InterGRU_optimizer.zero_grad()

#             # Forward pass through HANN
#             for reviews_ctr in range(len(training_batches)): # iter. through reviews

#                 IntraGRU_optimizer[reviews_ctr].zero_grad()

#                 current_batch = training_batches[reviews_ctr][batch_ctr]            
#                 input_variable, lengths, ratings = current_batch
#                 input_variable = input_variable.to(device)
#                 lengths = lengths.to(device)

#                 current_asins = torch.tensor(candidate_items[idx][batch_ctr]).to(device)
#                 current_reviewerIDs = torch.tensor(candidate_users[idx][batch_ctr]).to(device)

#                 outputs, intra_hidden, intra_attn_score = IntraGRU[reviews_ctr](input_variable, lengths, 
#                     current_asins, current_reviewerIDs)
                
#                 outputs = outputs.unsqueeze(0)

#                 if(isCatItemVec):
#                     # Concat. asin feature
#                     this_asins = training_item_batches[reviews_ctr][batch_ctr]
#                     this_asins = torch.tensor([val for val in this_asins]).to(device)
#                     this_asins = this_asins.unsqueeze(0)
#                 else:
#                     interInput_asin = None

#                 if(reviews_ctr == 0):
#                     interInput = outputs
#                     if(isCatItemVec):
#                         interInput_asin = this_asins
#                 else:
#                     interInput = torch.cat((interInput, outputs) , 0) 
#                     if(isCatItemVec):
#                         interInput_asin = torch.cat((interInput_asin, this_asins) , 0) 


#             outputs, intra_hidden, inter_attn_score  = InterGRU(interInput, interInput_asin, current_asins, current_reviewerIDs)
#             outputs = outputs.squeeze(1)
            
#             # Caculate loss 
#             current_labels = torch.tensor(training_batch_labels[idx][batch_ctr]).to(device)
#             # current_labels = torch.tensor(training_batch_labels[idx][batch_ctr])

#             err = (outputs*(5-1)+1) - current_labels
#             loss = torch.mul(err, err)
#             loss = torch.mean(loss, dim=0)
            
#             # Perform backpropatation
#             loss.backward()

#             # Clip gradients: gradients are modified in place
#             for reviews_ctr in range(len(training_batches)):            
#                 _ = nn.utils.clip_grad_norm_(IntraGRU[reviews_ctr].parameters(), opt.clip)
#             _ = nn.utils.clip_grad_norm_(InterGRU.parameters(), opt.clip)

#             # Adjust model weights
#             for reviews_ctr in range(len(training_batches)):
#                 IntraGRU_optimizer[reviews_ctr].step()
#             InterGRU_optimizer.step()

#             epoch_loss += loss

#     return epoch_loss

# # Train model
# def Train(myVoc, table, training_batches, training_item_batches, candidate_items, candidate_users, training_batch_labels, 
#      directory, TrainEpoch=100, latentK=32, hidden_size = 300, intra_method ='dualFC', inter_method='dualFC',
#      learning_rate = 0.00001, dropout=0, isStoreModel=False, isStoreCheckPts=False, WriteTrainLoss=False, store_every = 2, use_pretrain_item= False, 
#      isCatItemVec= True, pretrain_wordVec=None):
    
#     # Get asin and reviewerID from file
#     asin, reviewerID = data_preprocess._read_asin_reviewer(table='clothing_')

#     # Initialize textual embeddings
#     if(pretrain_wordVec != None):
#         embedding = pretrain_wordVec
#     else:
#         embedding = nn.Embedding(myVoc.num_words, hidden_size)

#     # Initialize asin/reviewer embeddings
#     if(use_pretrain_item):
#         asin_embedding = torch.load(R'PretrainingEmb/item_embedding_fromGRU.pth')
#     else:
#         asin_embedding = nn.Embedding(len(asin), hidden_size)
#     reviewerID_embedding = nn.Embedding(len(reviewerID), hidden_size)    
    
#     # Initialize IntraGRU models and optimizers
#     IntraGRU = list()
#     IntraGRU_optimizer = list()

#     # Initialize IntraGRU optimizers groups
#     intra_scheduler = list()

#     # Append GRU model asc
#     for idx in range(opt.num_of_reviews):    
#         IntraGRU.append(IntraReviewGRU(hidden_size, embedding, asin_embedding, reviewerID_embedding,  
#             latentK = latentK, method=intra_method))
#         # Use appropriate device
#         IntraGRU[idx] = IntraGRU[idx].to(device)
#         IntraGRU[idx].train()

#         # Initialize optimizers
#         IntraGRU_optimizer.append(optim.AdamW(IntraGRU[idx].parameters(), 
#                 lr=learning_rate, weight_decay=0.001)
#             )
        
#         # Assuming optimizer has two groups.
#         intra_scheduler.append(optim.lr_scheduler.StepLR(IntraGRU_optimizer[idx], 
#             step_size=20, gamma=0.3))

    
#     # Initialize InterGRU models
#     InterGRU = HANN(hidden_size, embedding, asin_embedding, reviewerID_embedding,
#             n_layers=1, dropout=dropout, latentK = latentK, isCatItemVec=isCatItemVec , net_type=opt.net_type, method=inter_method)

#     # Use appropriate device
#     InterGRU = InterGRU.to(device)
#     InterGRU.train()
#     # Initialize IntraGRU optimizers    
#     InterGRU_optimizer = optim.AdamW(InterGRU.parameters(), 
#             lr=learning_rate, weight_decay=0.001)

#     # Assuming optimizer has two groups.
#     inter_scheduler = optim.lr_scheduler.StepLR(InterGRU_optimizer, 
#         step_size=10, gamma=0.3)

#     print('Models built and ready to go!')

#     for Epoch in range(TrainEpoch):
#         # Run a training iteration with batch
#         group_loss = trainIteration(IntraGRU, InterGRU, IntraGRU_optimizer, InterGRU_optimizer, training_batches, training_item_batches, 
#             candidate_items, candidate_users, training_batch_labels, isCatItemVec=isCatItemVec)

#         inter_scheduler.step()
#         for idx in range(opt.num_of_reviews):
#             intra_scheduler[idx].step()

#         num_of_iter = len(training_batches[0])*len(training_batch_labels)
#         current_loss_average = group_loss/num_of_iter
#         print('Epoch:{}\tSE:{}\t'.format(Epoch, current_loss_average))

#         if(Epoch % store_every == 0 and isStoreModel):
#             torch.save(InterGRU, R'{}/Model/InterGRU_epoch{}'.format(opt.save_dir, Epoch))
#             for idx__, IntraGRU__ in enumerate(IntraGRU):
#                 torch.save(IntraGRU__, R'{}/Model/IntraGRU_idx{}_epoch{}'.format(opt.save_dir, idx__, Epoch))
                    
#         if WriteTrainLoss:
#             with open(R'{}/Loss/TrainingLoss.txt'.format(opt.save_dir),'a') as file:
#                 file.write('Epoch:{}\tSE:{}\n'.format(Epoch, current_loss_average))  

#         # Save checkpoint
#         if (Epoch % store_every == 0 and isStoreCheckPts):
#             # Store intra-GRU model
#             for idx__, IntraGRU__ in enumerate(IntraGRU):
#                 state = {
#                     'epoch': Epoch,
#                     'num_of_review': idx__,
#                     'intra{}'.format(idx__): IntraGRU__.state_dict(),
#                     'intra{}_opt'.format(idx__): IntraGRU_optimizer[idx__].state_dict(),
#                     'train_loss': current_loss_average,
#                     'voc_dict': myVoc.__dict__,
#                     'embedding': embedding.state_dict()
#                 }
#                 torch.save(state, R'{}/checkpts/IntraGRU_idx{}_epoch{}'.format(opt.save_dir, idx__, Epoch))
            
#             # Store inter-GRU model
#             state = {
#                 'epoch': Epoch,
#                 'inter': InterGRU.state_dict(),
#                 'inter_opt': InterGRU_optimizer.state_dict(),
#                 'train_loss': current_loss_average,
#                 'voc_dict': myVoc.__dict__,
#                 'embedding': embedding.state_dict()
#             }
#             torch.save(state, R'{}/checkpts/InterGRU_epoch{}'.format(opt.save_dir, Epoch))




# def evaluate(IntraGRU, InterGRU, training_batches, training_asin_batches, validate_batch_labels, validate_asins, validate_reviewerIDs, 
#     isCatItemVec=False, isWriteAttn=False, userObj=None):
    
#     group_loss = 0
#     # Voutput = visulizeOutput.WriteSentenceHeatmap(opt.save_dir, opt.num_of_reviews)
#     AttnVisualize = Visualization(opt.save_dir, opt.visulize_attn_epoch, opt.num_of_reviews)

#     # for batch_ctr in tqdm.tqdm(range(len(training_batches[0]))): #how many batches
#     for batch_ctr in range(len(training_batches[0])): #how many batches
#         for idx in range(len(validate_batch_labels)):
#             for reviews_ctr in range(len(training_batches)): #loop review 1 to 5
                
#                 current_batch = training_batches[reviews_ctr][batch_ctr]
                
#                 input_variable, lengths, ratings = current_batch
#                 input_variable = input_variable.to(device)
#                 lengths = lengths.to(device)

#                 current_asins = torch.tensor(validate_asins[idx][batch_ctr]).to(device)
#                 current_reviewerIDs = torch.tensor(validate_reviewerIDs[idx][batch_ctr]).to(device)
        
#                 # Concat. asin feature
#                 this_asins = training_asin_batches[reviews_ctr][batch_ctr]
#                 this_asins = torch.tensor([val for val in this_asins]).to(device)
#                 this_asins = this_asins.unsqueeze(0)

#                 with torch.no_grad():
#                     outputs, intra_hidden, intra_attn_score = IntraGRU[reviews_ctr](input_variable, lengths, 
#                         current_asins, current_reviewerIDs)
#                     outputs = outputs.unsqueeze(0)

#                     if(reviews_ctr == 0):
#                         interInput = outputs
#                         interInput_asin = this_asins
#                     else:
#                         interInput = torch.cat((interInput, outputs) , 0) 
#                         interInput_asin = torch.cat((interInput_asin, this_asins) , 0) 

#                 # Writing Intra-attention weight to .html file
#                 if(isWriteAttn):
#                     if(opt.net_type=='user_base'):
#                         current_candidates = current_reviewerIDs
#                     elif(opt.net_type=='item_base'):
#                         current_candidates = current_asins
#                         pass
                    
#                     for index_ , candidateObj_ in enumerate(current_candidates):

#                         intra_attn_wts = intra_attn_score[:,index_].squeeze(1).tolist()
#                         word_indexes = input_variable[:,index_].tolist()

#                         # Voutput.js(intra_attn_wts, word_indexes, voc.index2word, reviews_ctr, fname='{}@{}'.format( userObj.index2reviewerID[user_.item()], reviews_ctr))
                        
#                         sentence, weights = AttnVisualize.wdIndex2sentences(word_indexes, voc.index2word, intra_attn_wts)
#                         AttnVisualize.createHTML(sentence, weights, reviews_ctr, 
#                             fname='{}@{}'.format( userObj.index2reviewerID[candidateObj_.item()], reviews_ctr)
#                             )
                            
#             with torch.no_grad():
#                 outputs, intra_hidden, inter_attn_score  = InterGRU(interInput, interInput_asin, current_asins, current_reviewerIDs)
#                 outputs = outputs.squeeze(1)

#             # Writing Inter-attention weight to .txt file
#             if(isWriteAttn):
#                 if(opt.net_type=='user_base'):
#                     current_candidates = current_reviewerIDs
#                 elif(opt.net_type=='item_base'):
#                     current_candidates = current_asins
#                     pass                

#                 for index_ , candidateObj_ in enumerate(current_candidates):
#                     inter_attn_wts = inter_attn_score.squeeze(2)[:,index_].tolist()
#                     with open('{}/VisualizeAttn/inter.txt'.format(opt.save_dir), 'a') as file:
#                         file.write("=================================\nCandidateObj: {}\n".
#                             format(userObj.index2reviewerID[candidateObj_.item()]))
#                         for index__, val in enumerate(inter_attn_wts):
#                             file.write('{} ,{}\n'.format(index__, val))           
            
#             current_labels = torch.tensor(validate_batch_labels[idx][batch_ctr]).to(device)

#             err = (outputs*(5-1)+1) - current_labels
#             loss = torch.mul(err, err)
#             loss = torch.mean(loss, dim=0)


#             loss = torch.sqrt(loss)

#             group_loss += loss

#     num_of_iter = len(training_batches[0])*len(validate_batch_labels)
#     RMSE = group_loss/num_of_iter

#     return RMSE





        # if(False):
        #     # Start to train
        #     Train(voc, opt.selectTable, training_batches, training_asin_batches, candidate_asins, candidate_reviewerIDs, training_batch_labels, 
        #         opt.save_dir, TrainEpoch=opt.epoch, latentK=opt.latentK, hidden_size=opt.hidden, intra_method=opt.intra_attn_method , inter_method=opt.inter_attn_method,
        #         learning_rate = opt.lr, dropout=opt.dropout, isStoreModel=True, WriteTrainLoss=True, store_every = opt.save_model_freq, 
        #         use_pretrain_item=False, isCatItemVec=True, pretrain_wordVec=pretrain_wordVec)
        
        # else: