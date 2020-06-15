import torch
import torch.nn as nn
from torch import optim

import tqdm
import random
from utils.model import IntraReviewGRU, HANN, HANN_i, HANN_u, MultiFC, DecoderGRU, HANN_new
from utils.setup import train_test_setup
from utils._wtensorboard import _Tensorboard
from visualization.attention_visualization import Visualization

import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

class RatingRegresstion(train_test_setup):
    def __init__(self, device, net_type, save_dir, voc, prerocess, 
        training_epoch=100, latent_k=32, batch_size=40, hidden_size=300, clip=50,
        num_of_reviews = 5, 
        intra_method='dualFC', inter_method='dualFC', 
        learning_rate=0.00001, dropout=0,
        setence_max_len=50):
        
        super(RatingRegresstion, self).__init__(device, net_type, save_dir, voc, prerocess, training_epoch, latent_k, batch_size, hidden_size, clip, num_of_reviews, intra_method, inter_method, learning_rate, dropout, setence_max_len)
        
        self._tesorboard = _Tensorboard(self.save_dir + '/tensorboard')
        self.use_sparsity_review = False

        pass

    def set_candidate_object(self, userObj, itemObj):
        self.itemObj = itemObj
        self.userObj = userObj
        pass

    def set_training_review_rating(self, training_review_rating, correspond_review_rating):
        self.training_review_rating = training_review_rating
        self.training_correspond_review_rating = correspond_review_rating
        pass

    def set_testing_review_rating(self, testing_review_rating, correspond_review_rating):
        self.testing_review_rating = testing_review_rating
        self.testing_correspond_review_rating = correspond_review_rating
        pass

    def set_can2sparsity(self, can2sparsity):
        self.can2sparsity = can2sparsity
        self.use_sparsity_review = True
        pass

    def set_testing_batches(self, testing_batches, testing_external_memorys, testing_batch_labels, testing_asins, testing_reviewerIDs):
        self.testing_batches = testing_batches
        self.testing_external_memorys = testing_external_memorys
        self.testing_batch_labels = testing_batch_labels
        self.testing_asins = testing_asins
        self.testing_reviewerIDs = testing_reviewerIDs
        pass
    
    def set_testing_correspond_batches(self, testing_correspond_batches):
        self.testing_correspond_batches = testing_correspond_batches
        pass

    def set_correspond_batches(self, correspond_batches):
        self.correspond_batches = correspond_batches
        pass

    def set_correspond_external_memorys(self, external_memorys):
        self.correspond_external_memorys = external_memorys
        pass

    def set_correspond_num_of_reviews(self, num_of_reviews):
        self.correspond_num_of_reviews = num_of_reviews
        pass

    def set_sparsity(self, num_of_review):
        self._sparsity_review = num_of_review

    def _train_iteration(self, IntraGRU, InterGRU, IntraGRU_optimizer, InterGRU_optimizer, 
        training_batches, external_memorys, candidate_items, candidate_users, training_batch_labels, 
        isCatItemVec=False,concat_rating=False):
        """ Training each iteraction"""

        # Initialize this epoch loss
        epoch_loss = 0

        for batch_ctr in tqdm.tqdm(range(len(training_batches[0]))): # amount of batches
            # Run multiple label for training 
            for idx in range(len(training_batch_labels)):

                InterGRU_optimizer.zero_grad()

                # Forward pass through HANN
                for reviews_ctr in range(len(training_batches)): # iter. through reviews

                    IntraGRU_optimizer[reviews_ctr].zero_grad()

                    current_batch = training_batches[reviews_ctr][batch_ctr]            
                    input_variable, lengths, ratings = current_batch
                    input_variable = input_variable.to(self.device)
                    lengths = lengths.to(self.device)

                    current_asins = torch.tensor(candidate_items[idx][batch_ctr]).to(self.device)
                    current_reviewerIDs = torch.tensor(candidate_users[idx][batch_ctr]).to(self.device)

                    outputs, intra_hidden, intra_attn_score = IntraGRU[reviews_ctr](input_variable, lengths, 
                        current_asins, current_reviewerIDs)
                    
                    outputs = outputs.unsqueeze(0)

                    # if(isCatItemVec):
                    #     # Concat. asin feature
                    #     this_asins = external_memorys[reviews_ctr][batch_ctr]
                    #     this_asins = torch.tensor([val for val in this_asins]).to(self.device)
                    #     this_asins = this_asins.unsqueeze(0)
                    # else:
                    #     interInput_asin = None

                    if(concat_rating):
                        this_rating = self.training_review_rating[reviews_ctr][batch_ctr]
                        
                        _encode_rating = self._rating_to_onehot(this_rating)
                        _encode_rating = torch.tensor(_encode_rating).to(self.device)
                        _encode_rating = _encode_rating.unsqueeze(0)
                        pass
                    else:
                        inter_intput_rating =None       

                    
                    if(reviews_ctr == 0):
                        interInput = outputs
                        if(concat_rating):
                            inter_intput_rating = _encode_rating                        
                        # if(isCatItemVec):
                        #     interInput_asin = this_asins
                    else:
                        interInput = torch.cat((interInput, outputs) , 0) 
                        
                        # if(isCatItemVec):
                        #     interInput_asin = torch.cat((interInput_asin, this_asins) , 0)                         

                        if(concat_rating):
                            inter_intput_rating = torch.cat(
                                (inter_intput_rating, _encode_rating) , 0
                                ) 

                interInput_asin = None
                outputs, intra_hidden, inter_attn_score, context_vector  = InterGRU(
                    interInput, 
                    interInput_asin, 
                    current_asins, 
                    current_reviewerIDs,
                    review_rating = inter_intput_rating
                    )
                outputs = outputs.squeeze(1)

                # Caculate loss 
                current_labels = torch.tensor(training_batch_labels[idx][batch_ctr]).to(self.device)

                err = (outputs*(5-1)+1) - current_labels
                loss = torch.mul(err, err)
                loss = torch.mean(loss, dim=0)
                
                # Perform backpropatation
                loss.backward()

                # Clip gradients: gradients are modified in place
                for reviews_ctr in range(len(training_batches)):            
                    _ = nn.utils.clip_grad_norm_(IntraGRU[reviews_ctr].parameters(), self.clip)
                _ = nn.utils.clip_grad_norm_(InterGRU.parameters(), self.clip)

                # Adjust model weights
                for reviews_ctr in range(len(training_batches)):
                    IntraGRU_optimizer[reviews_ctr].step()
                InterGRU_optimizer.step()

                epoch_loss += loss

        return epoch_loss

    def train(self, select_table, isStoreModel=False, WriteTrainLoss=False, 
            store_every = 2, use_pretrain_item= False, 
            isCatItemVec=False, concat_rating=False, 
            pretrain_wordVec=None):
        
        asin, reviewerID = self._get_asin_reviewer()
        # Initialize textual embeddings
        if(pretrain_wordVec != None):
            embedding = pretrain_wordVec
        else:
            embedding = nn.Embedding(self.voc.num_words, self.hidden_size)


        # Initialize asin/reviewer embeddings
        if(use_pretrain_item):
            asin_embedding = torch.load(R'PretrainingEmb/item_embedding_fromGRU.pth')
        else:
            asin_embedding = nn.Embedding(len(asin), self.hidden_size)
        reviewerID_embedding = nn.Embedding(len(reviewerID), self.hidden_size)   


        # Initialize IntraGRU models and optimizers
        IntraGRU = list()
        IntraGRU_optimizer = list()

        # Initialize IntraGRU optimizers groups
        intra_scheduler = list()

        # Append GRU model asc
        for idx in range(self.num_of_reviews):    
            IntraGRU.append(
                IntraReviewGRU(
                    self.hidden_size, 
                    embedding, 
                    asin_embedding, 
                    reviewerID_embedding,  
                    latentK = self.latent_k, 
                    method=self.intra_method)
                    )
            # Use appropriate device
            IntraGRU[idx] = IntraGRU[idx].to(self.device)
            IntraGRU[idx].train()

            # Initialize optimizers
            IntraGRU_optimizer.append(optim.AdamW(IntraGRU[idx].parameters(), 
                    lr=self.learning_rate, weight_decay=0.001)
                )
            
            # Assuming optimizer has two groups.
            intra_scheduler.append(optim.lr_scheduler.StepLR(IntraGRU_optimizer[idx], 
                step_size=20, gamma=0.3))

        # Initialize InterGRU models
        InterGRU = HANN_new(
            self.hidden_size, 
            embedding, 
            asin_embedding, 
            reviewerID_embedding,
            n_layers=1, 
            dropout=self.dropout, 
            latentK = self.latent_k, 
            isCatItemVec=isCatItemVec , 
            concat_rating= concat_rating,
            netType=self.net_type, 
            method=self.inter_method
            )

        # Use appropriate device
        InterGRU = InterGRU.to(self.device)
        InterGRU.train()

        # Initialize IntraGRU optimizers    
        InterGRU_optimizer = optim.AdamW(
            InterGRU.parameters(), 
            lr=self.learning_rate, 
            weight_decay=0.001
            )

        # Assuming optimizer has two groups.
        inter_scheduler = optim.lr_scheduler.StepLR(InterGRU_optimizer, 
            step_size=10, gamma=0.3)

        print('Models built and ready to go!')        

        for Epoch in range(self.training_epoch):
            # Run a training iteration with batch
            group_loss = self._train_iteration(
                IntraGRU, 
                InterGRU, 
                IntraGRU_optimizer, 
                InterGRU_optimizer, 
                self.training_batches, 
                self.external_memorys, 
                self.candidate_items, 
                self.candidate_users, 
                self.training_batch_labels, 
                isCatItemVec=isCatItemVec, 
                concat_rating=concat_rating
                )

            RMSE = self.evaluate(
                IntraGRU, 
                InterGRU, 
                isWriteAttn=False,
                isCatItemVec=isCatItemVec, 
                concat_rating=concat_rating
                )

            inter_scheduler.step()
            for idx in range(self.num_of_reviews):
                intra_scheduler[idx].step()

            num_of_iter = len(self.training_batches[0])*len(self.training_batch_labels)
            current_loss_average = group_loss/num_of_iter
            print('Epoch:{}\tSE:{}\t'.format(Epoch, current_loss_average))
            print('Epoch:{}\tMSE:{}\t'.format(Epoch, RMSE))
    

            if(Epoch % store_every == 0 and isStoreModel):
                torch.save(InterGRU, R'{}/Model/InterGRU_epoch{}'.format(self.save_dir, Epoch))
                for idx__, IntraGRU__ in enumerate(IntraGRU):
                    torch.save(IntraGRU__, R'{}/Model/IntraGRU_idx{}_epoch{}'.format(self.save_dir, idx__, Epoch))
                        
            if WriteTrainLoss:
                with open(R'{}/Loss/TrainingLoss.txt'.format(self.save_dir),'a') as file:
                    file.write('Epoch:{}\tSE:{}\n'.format(Epoch, current_loss_average))  

                with open(R'{}/Loss/TestingLoss.txt'.format(self.save_dir),'a') as file:
                    file.write('Epoch:{}\tRMSE:{}\n'.format(Epoch, RMSE))

        pass

    def evaluate(self, IntraGRU, InterGRU, isCatItemVec=False, concat_rating=False, isWriteAttn=False, userObj=None, visulize_attn_epoch=0):
        
        group_loss = 0

        for batch_ctr in range(len(self.testing_batches[0])): #how many batches
            for idx in range(len(self.testing_batch_labels)):
                for reviews_ctr in range(len(self.testing_batches)): #loop review 1 to 5
                    
                    current_batch = self.testing_batches[reviews_ctr][batch_ctr]
                    
                    input_variable, lengths, ratings = current_batch
                    input_variable = input_variable.to(self.device)
                    lengths = lengths.to(self.device)

                    current_asins = torch.tensor(self.testing_asins[idx][batch_ctr]).to(self.device)
                    current_reviewerIDs = torch.tensor(self.testing_reviewerIDs[idx][batch_ctr]).to(self.device)
            
                    if(concat_rating):
                        this_rating = self.testing_review_rating[reviews_ctr][batch_ctr]
                        
                        _encode_rating = self._rating_to_onehot(this_rating)
                        _encode_rating = torch.tensor(_encode_rating).to(self.device)
                        _encode_rating = _encode_rating.unsqueeze(0)
                        pass
                    else:
                        inter_intput_rating =None

                    with torch.no_grad():
                        outputs, intra_hidden, intra_attn_score = IntraGRU[reviews_ctr](
                            input_variable, 
                            lengths, 
                            current_asins, 
                            current_reviewerIDs
                            )
                        outputs = outputs.unsqueeze(0)

                        if(reviews_ctr == 0):
                            interInput = outputs
                            if(concat_rating):
                                inter_intput_rating = _encode_rating                        

                        else:
                            interInput = torch.cat((interInput, outputs) , 0) 
                            if(concat_rating):
                                inter_intput_rating = torch.cat(
                                    (inter_intput_rating, _encode_rating) , 0
                                    ) 
                
                interInput_asin = None
                with torch.no_grad():
                    outputs, intra_hidden, inter_attn_score, context_vector = InterGRU(
                        interInput, 
                        interInput_asin, 
                        current_asins, 
                        current_reviewerIDs,
                        review_rating = inter_intput_rating
                        )
                    outputs = outputs.squeeze(1)
        
                
                current_labels = torch.tensor(self.testing_batch_labels[idx][batch_ctr]).to(self.device)

                err = (outputs*(5-1)+1) - current_labels
                loss = torch.mul(err, err)
                loss = torch.mean(loss, dim=0)
                loss = torch.sqrt(loss)

                group_loss += loss

        num_of_iter = len(self.testing_batches[0])*len(self.testing_batch_labels)
        RMSE = group_loss/num_of_iter

        return RMSE

    def _hybird_train_iteration(self, 
        intra_GRU, inter_GRU, intra_GRU_optimizer, inter_GRU_optimizer,
        correspond_intra_GRU, correspond_inter_GRU, correspond_intra_GRU_optimizer, correspond_inter_GRU_optimizer, 
        MFC, MFC_optimizer,
        training_batches, external_memorys, candidate_items, 
        candidate_users, training_batch_labels, 
        correspond_batches,
        isCatItemVec=False,
        concat_rating = False
        ):

        """ Training each iteraction"""

        # Initialize this epoch loss
        epoch_loss = 0

        for batch_ctr in tqdm.tqdm(range(len(training_batches[0]))): # amount of batches
            # Run multiple label for training 
            for idx in range(len(training_batch_labels)):
                
                # Initialize optimizer
                inter_GRU_optimizer.zero_grad()
                correspond_inter_GRU_optimizer.zero_grad()
                MFC_optimizer.zero_grad()

                """
                Forward pass through base net model
                """
                for reviews_ctr in range(len(training_batches)): # iter. through reviews
                    
                    # Initialize optimizer
                    intra_GRU_optimizer[reviews_ctr].zero_grad()

                    # tmp = 

                    current_batch = training_batches[reviews_ctr][batch_ctr]
                    input_variable, lengths, ratings = current_batch
                    input_variable = input_variable.to(self.device)
                    lengths = lengths.to(self.device)

                    current_asins = torch.tensor(candidate_items[idx][batch_ctr]).to(self.device)
                    current_reviewerIDs = torch.tensor(candidate_users[idx][batch_ctr]).to(self.device)

                    base_net_outputs, intra_hidden, intra_attn_score = intra_GRU[reviews_ctr](
                        input_variable, 
                        lengths, 
                        current_asins, 
                        current_reviewerIDs
                        )
                    
                    base_net_outputs = base_net_outputs.unsqueeze(0)

                    if(isCatItemVec):
                        # Concat. asin feature
                        this_asins = external_memorys[reviews_ctr][batch_ctr]
                        this_asins = torch.tensor([val for val in this_asins]).to(self.device)
                        this_asins = this_asins.unsqueeze(0)
                    else:
                        interInput_asin = None

                    if(reviews_ctr == 0):
                        interInput = base_net_outputs
                        if(isCatItemVec):
                            interInput_asin = this_asins
                    else:
                        interInput = torch.cat((interInput, base_net_outputs) , 0) 
                        if(isCatItemVec):
                            interInput_asin = torch.cat((interInput_asin, this_asins) , 0) 
                    

                    if(concat_rating):
                        this_rating = self.training_review_rating[reviews_ctr][batch_ctr]

                        _encode_rating = self._rating_to_onehot(this_rating)
                        _encode_rating = torch.tensor(_encode_rating).to(self.device)
                        _encode_rating = _encode_rating.unsqueeze(0)
                        pass
                    else:
                        inter_intput_rating =None           


                    if(reviews_ctr == 0):
                        interInput = base_net_outputs
                        if(concat_rating):
                            inter_intput_rating = _encode_rating
                    else:
                        if(concat_rating):
                            inter_intput_rating = torch.cat(
                                (inter_intput_rating, _encode_rating) , 0
                                ) 
                                    
                base_net_outputs, intra_hidden, inter_attn_score  = inter_GRU(
                    interInput, 
                    interInput_asin, 
                    current_asins, 
                    current_reviewerIDs,
                    review_rating = inter_intput_rating
                    )


                # base_net_outputs = base_net_outputs.squeeze(1)

                """
                Forward pass through correspond net model
                """
                for reviews_ctr in range(self.correspond_num_of_reviews): # iter. through reviews
                    
                    # Initialize optimizer
                    correspond_intra_GRU_optimizer[reviews_ctr].zero_grad()

                    current_batch = correspond_batches[reviews_ctr][batch_ctr]            
                    input_variable, lengths, ratings = current_batch
                    input_variable = input_variable.to(self.device)
                    lengths = lengths.to(self.device)

                    correspond_current_asins = torch.tensor(candidate_items[idx][batch_ctr]).to(self.device)
                    correspond_current_reviewerIDs = torch.tensor(candidate_users[idx][batch_ctr]).to(self.device)

                    correspond_net_outputs, intra_hidden, intra_attn_score = correspond_intra_GRU[reviews_ctr](
                        input_variable, 
                        lengths, 
                        correspond_current_asins, 
                        correspond_current_reviewerIDs
                        )
                    
                    correspond_net_outputs = correspond_net_outputs.unsqueeze(0)

                    if(isCatItemVec):
                        # Concat. asin feature
                        this_asins = external_memorys[reviews_ctr][batch_ctr]
                        this_asins = torch.tensor([val for val in this_asins]).to(self.device)
                        this_asins = this_asins.unsqueeze(0)
                    else:
                        interInput_asin = None

                    if(reviews_ctr == 0):
                        interInput = correspond_net_outputs
                        if(isCatItemVec):
                            interInput_asin = this_asins
                    else:
                        interInput = torch.cat((interInput, correspond_net_outputs) , 0) 
                        if(isCatItemVec):
                            interInput_asin = torch.cat((interInput_asin, this_asins) , 0) 

                    if(concat_rating):
                        this_rating = self.training_correspond_review_rating[reviews_ctr][batch_ctr]
                        
                        _encode_rating = self._rating_to_onehot(this_rating)
                        _encode_rating = torch.tensor(_encode_rating).to(self.device)
                        _encode_rating = _encode_rating.unsqueeze(0)
                        pass
                    else:
                        inter_intput_rating =None           


                    if(reviews_ctr == 0):
                        interInput = correspond_net_outputs
                        if(concat_rating):
                            inter_intput_rating = _encode_rating
                    else:
                        if(concat_rating):
                            inter_intput_rating = torch.cat(
                                (inter_intput_rating, _encode_rating) , 0
                                )                                

                correspond_net_outputs, intra_hidden, inter_attn_score  = correspond_inter_GRU(
                    interInput, 
                    interInput_asin, 
                    current_asins, 
                    current_reviewerIDs,
                    review_rating = inter_intput_rating
                    )

                correspond_net_outputs = correspond_net_outputs.squeeze(1)



                # Writing Intra-attention weight to .html file
                if(not True and batch_ctr > 160):
                    
                    """Only considerate user attention"""
                    # current_candidates = correspond_current_reviewerIDs
                    # current_asins
                    # current_reviewerIDs                    
                    
                    for index_ , candidateObj_ in enumerate(current_asins):

                        
                        _item = self.itemObj.index2asin[current_asins[index_].item()]
                        _user = self.userObj.index2reviewerID[current_reviewerIDs[index_].item()]
            
                        word_indexes = input_variable[:,index_].tolist()
                        words = [self.voc.index2word[index] for index in word_indexes if self.voc.index2word[index] != 'PAD']
                        sentence = [" ".join(words)]

                        stop = 1
  

                """
                Forward pass through final prediction net model
                """                
                finalOut = MFC(
                    base_net_outputs, 
                    correspond_net_outputs, 
                    current_asins, 
                    current_reviewerIDs
                    )
                finalOut = finalOut.squeeze(1)
                 
                
                opt_method = 'se'
                if(opt_method == 'se'):
                    # Caculate loss 
                    current_labels = torch.tensor(training_batch_labels[idx][batch_ctr]).to(self.device)    # grond truth

                    err = (finalOut*(5-1)+1) - current_labels
                    loss = torch.mul(err, err)
                    loss = torch.mean(loss, dim=0)

                elif(opt_method == 'nll'):
                    loss = 0
                    current_labels = torch.tensor(training_batch_labels[idx][batch_ctr]).to(self.device)    # grond truth
                    for index_, label_ in enumerate(current_labels):
                        true_lb = int(label_.item())
                        
                        # Calculate nll
                        _ = torch.log(
                            finalOut[index_][true_lb-1]
                            )

                        loss += _                
                    loss = (-1) * loss
                
                # Perform backpropatation
                loss.backward()

                # Clip gradients: gradients are modified in place
                for reviews_ctr in range(len(training_batches)):            
                    _ = nn.utils.clip_grad_norm_(intra_GRU[reviews_ctr].parameters(), self.clip)
                _ = nn.utils.clip_grad_norm_(inter_GRU.parameters(), self.clip)

                # Adjust model weights
                for reviews_ctr in range(len(training_batches)):
                    intra_GRU_optimizer[reviews_ctr].step()
                for reviews_ctr in range(4):
                    correspond_intra_GRU_optimizer[reviews_ctr].step()

                inter_GRU_optimizer.step()
                correspond_inter_GRU_optimizer.step()
                MFC_optimizer.step()

                epoch_loss += loss

        return epoch_loss

    def hybird_train(self, select_table, isStoreModel=False, isStoreCheckPts=False, WriteTrainLoss=False, store_every = 2, 
            use_pretrain_item= False, isCatItemVec= True, pretrain_wordVec=None, concat_rating=False, epoch_to_store = 0):
        
        asin, reviewerID = self._get_asin_reviewer()
        # Initialize textual embeddings
        if(pretrain_wordVec != None):
            embedding = pretrain_wordVec
        else:
            embedding = nn.Embedding(self.voc.num_words, self.hidden_size)

        # Initialize asin/reviewer embeddings
        if(use_pretrain_item):
            asin_embedding = torch.load(R'PretrainingEmb/item_embedding_fromGRU.pth')
        else:
            asin_embedding = nn.Embedding(len(asin), self.hidden_size)
        reviewerID_embedding = nn.Embedding(len(reviewerID), self.hidden_size)   


        #---------------------------------------- Base net model construction start ------------------------------------------#

        # Initialize intra_GRU models and optimizers
        intra_GRU = list()
        intra_GRU_optimizer = list()

        # Initialize intra_GRU optimizers groups
        intra_scheduler = list()

        # Append GRU model asc
        for idx in range(self.num_of_reviews):    
            intra_GRU.append(
                IntraReviewGRU(
                    self.hidden_size, 
                    embedding, asin_embedding, 
                    reviewerID_embedding,  
                    latentK = self.latent_k, 
                    method=self.intra_method
                    )
                )
            # Use appropriate device
            intra_GRU[idx] = intra_GRU[idx].to(self.device)
            intra_GRU[idx].train()

            # Initialize optimizers
            intra_GRU_optimizer.append(
                optim.AdamW(
                    intra_GRU[idx].parameters(), 
                    lr=self.learning_rate, 
                    weight_decay=0.001
                    )
                )
            
            # Assuming optimizer has two groups.
            intra_scheduler.append(
                optim.lr_scheduler.StepLR(
                    intra_GRU_optimizer[idx], 
                    step_size=20, 
                    gamma=0.3
                    )
                )

        
        # Initialize inter_GRU models
        inter_GRU = HANN_i(
            self.hidden_size, 
            embedding, 
            asin_embedding, 
            reviewerID_embedding,
            n_layers=1, 
            dropout=self.dropout, 
            latentK = self.latent_k, 
            isCatItemVec=isCatItemVec , 
            concat_rating = concat_rating,
            netType=self.net_type, 
            method=self.inter_method
            )

        # Use appropriate device
        inter_GRU = inter_GRU.to(self.device)
        inter_GRU.train()

        # Initialize intra_GRU optimizers    
        inter_GRU_optimizer = optim.AdamW(
            inter_GRU.parameters(), 
            lr=self.learning_rate, 
            weight_decay=0.001
            )

        # Assuming optimizer has two groups.
        inter_scheduler = optim.lr_scheduler.StepLR(
            inter_GRU_optimizer, 
            step_size=10, 
            gamma=0.3
            )

        #---------------------------------------- Base net model construction complete. ------------------------------------------#


        #---------------------------------------- Correspond net model construction start ------------------------------------------#

        # Initialize intra_GRU models and optimizers
        correspond_intra_GRU = list()
        correspond_intra_GRU_optimizer = list()

        # Initialize intra_GRU optimizers groups
        correspond_intra_scheduler = list()

        # Append GRU model asc
        for idx in range(self.correspond_num_of_reviews):    
            correspond_intra_GRU.append(
                IntraReviewGRU(
                    self.hidden_size, 
                    embedding, 
                    asin_embedding, 
                    reviewerID_embedding,  
                    latentK = self.latent_k, 
                    method=self.intra_method
                )
            )

            # Use appropriate device
            correspond_intra_GRU[idx] = correspond_intra_GRU[idx].to(self.device)
            correspond_intra_GRU[idx].train()

            # Initialize optimizers
            correspond_intra_GRU_optimizer.append(
                optim.AdamW(
                    correspond_intra_GRU[idx].parameters(), 
                    lr=self.learning_rate, 
                    weight_decay=0.001
                    )
                )
            
            # Assuming optimizer has two groups.
            correspond_intra_scheduler.append(
                optim.lr_scheduler.StepLR(
                    correspond_intra_GRU_optimizer[idx], 
                    step_size=20, 
                    gamma=0.3
                    )
                )

        
        # Initialize inter_GRU models
        correspond_inter_GRU = HANN_u(
            self.hidden_size, 
            embedding, 
            asin_embedding, 
            reviewerID_embedding,
            n_layers=1, 
            dropout=self.dropout, 
            latentK = self.latent_k, 
            isCatItemVec=isCatItemVec ,
            concat_rating = concat_rating, 
            netType=self.net_type, 
            method=self.inter_method
            )

        # Use appropriate device
        correspond_inter_GRU = correspond_inter_GRU.to(self.device)
        correspond_inter_GRU.train()

        # Initialize intra_GRU optimizers    
        correspond_inter_GRU_optimizer = optim.AdamW(
            correspond_inter_GRU.parameters(), 
            lr=self.learning_rate, 
            weight_decay=0.001
            )

        # Assuming optimizer has two groups.
        correspond_inter_scheduler = optim.lr_scheduler.StepLR(
            correspond_inter_GRU_optimizer, 
            step_size=10, 
            gamma=0.3
            )

        #---------------------------------------- Correspond net model construction complete. ------------------------------------------#


        #---------------------------------------- Final  model construction start ------------------------------------------#

        MFC = MultiFC(self.hidden_size, asin_embedding, reviewerID_embedding, dropout=self.dropout, latentK = self.latent_k)
        # Use appropriate device
        MFC = MFC.to(self.device)


        # Initialize intra_GRU optimizers    
        MFC_optimizer = optim.AdamW(MFC.parameters(), lr=self.learning_rate, weight_decay=0.001)

        # Assuming optimizer has two groups.
        MFC_scheduler = optim.lr_scheduler.StepLR(MFC_optimizer, 
            step_size=10, gamma=0.3)

        #---------------------------------------- Base net model construction complete. ------------------------------------------#
        
        for Epoch in range(self.training_epoch):
            # Run a training iteration with batch
            group_loss = self._hybird_train_iteration(
                intra_GRU, inter_GRU, intra_GRU_optimizer, inter_GRU_optimizer, 
                correspond_intra_GRU, correspond_inter_GRU, correspond_intra_GRU_optimizer, correspond_inter_GRU_optimizer, 
                MFC, MFC_optimizer,
                self.training_batches, self.external_memorys, self.candidate_items, self.candidate_users, 
                self.training_batch_labels, 
                self.correspond_batches,
                isCatItemVec=isCatItemVec,
                concat_rating = concat_rating
                )                

            # Adjust optimizer group
            inter_scheduler.step()
            correspond_inter_scheduler.step()
            MFC_scheduler.step()
            for idx in range(self.correspond_num_of_reviews):
                correspond_intra_scheduler[idx].step
            for idx in range(self.num_of_reviews):
                intra_scheduler[idx].step()

            # Caculate epoch loss
            num_of_iter = len(self.training_batches[0])*len(self.training_batch_labels)
            current_loss_average = group_loss/num_of_iter
            print('Epoch:{}\tSE:{}\t'.format(Epoch, current_loss_average))


            RMSE, Accuracy, _ = self._hybird_evaluate(
                intra_GRU, inter_GRU, correspond_intra_GRU, correspond_inter_GRU, MFC,
                self.testing_batches, self.testing_external_memorys, self.testing_batch_labels, 
                self.testing_asins, self.testing_reviewerIDs,
                self.testing_correspond_batches, 
                isCatItemVec=isCatItemVec,
                concat_rating = concat_rating
                )
            print('Epoch:{}\tMSE:{}\tAccuracy:{}'.format(Epoch, RMSE, Accuracy))

            # Write confusion matrix
            plt.figure()
            self.plot_confusion_matrix(
                _, 
                classes = ['1pt', '2pt', '3pt', '4pt', '5pt'],
                normalize = not True,
                title = 'confusion matrix'
                )

            plt.savefig('{}/Loss/Confusion.Matrix/_{}.png'.format(
                self.save_dir,
                Epoch
            ))    


            if(Epoch % store_every == 0 and isStoreModel and Epoch >= epoch_to_store):
                torch.save(inter_GRU, R'{}/Model/InterGRU_epoch{}'.format(self.save_dir, Epoch))
                for idx__, IntraGRU__ in enumerate(intra_GRU):
                    torch.save(IntraGRU__, R'{}/Model/IntraGRU_idx{}_epoch{}'.format(self.save_dir, idx__, Epoch))

                torch.save(correspond_inter_GRU, R'{}/Model/correspond_InterGRU_epoch{}'.format(self.save_dir, Epoch))
                for idx__, correspond_IntraGRU__ in enumerate(correspond_intra_GRU):
                    torch.save(correspond_IntraGRU__, R'{}/Model/correspond_IntraGRU_idx{}_epoch{}'.format(self.save_dir, idx__, Epoch))                    

                torch.save(MFC, R'{}/Model/MFC_epoch{}'.format(self.save_dir, Epoch))
                        
            if WriteTrainLoss:
                with open(R'{}/Loss/TrainingLoss.txt'.format(self.save_dir),'a') as file:
                    file.write('Epoch:{}\tSE:{}\n'.format(Epoch, current_loss_average))  

                with open(R'{}/Loss/TestingLoss.txt'.format(self.save_dir),'a') as file:
                    file.write('Epoch:{}\tRMSE:{}\n'.format(Epoch, RMSE))  

                with open(R'{}/Loss/Accuracy.txt'.format(self.save_dir),'a') as file:
                    file.write('Epoch:{}\tAccuracy:{}\n'.format(Epoch, Accuracy))                      
        pass

    def set_ran_sparsity(self, _ran_sparsity=False, _reviews_be_chosen=None):
        self._ran_sparsity = _ran_sparsity
        self._reviews_be_chosen = _reviews_be_chosen
        pass

    def _hybird_evaluate(self, IntraGRU, InterGRU, correspond_intra_GRU, correspond_inter_GRU, MFC,
        training_batches, training_asin_batches, validate_batch_labels, validate_asins, validate_reviewerIDs, 
        correspond_batches,
        isCatItemVec=False, concat_rating=False,
        isWriteAttn=False, candidateObj=None, visulize_attn_epoch=0):
        
        group_loss = 0
        _accuracy = 0
        AttnVisualize = Visualization(self.save_dir, visulize_attn_epoch, self.num_of_reviews)
        
        true_label = list()
        predict_label = list()

        for batch_ctr in range(len(training_batches[0])): #how many batches
            for idx in range(len(validate_batch_labels)):
                for reviews_ctr in range(len(training_batches)): #loop review 1 to 5
                    
                    current_batch = training_batches[reviews_ctr][batch_ctr]
                    
                    input_variable, lengths, ratings = current_batch
                    input_variable = input_variable.to(self.device)
                    lengths = lengths.to(self.device)

                    current_asins = torch.tensor(validate_asins[idx][batch_ctr]).to(self.device)
                    current_reviewerIDs = torch.tensor(validate_reviewerIDs[idx][batch_ctr]).to(self.device)
            
                    # Concat. asin feature
                    this_asins = training_asin_batches[reviews_ctr][batch_ctr]
                    this_asins = torch.tensor([val for val in this_asins]).to(self.device)
                    this_asins = this_asins.unsqueeze(0)

                    with torch.no_grad():
                        outputs, intra_hidden, intra_attn_score = IntraGRU[reviews_ctr](
                            input_variable, lengths, 
                            current_asins, current_reviewerIDs)
                        outputs = outputs.unsqueeze(0)  # size : [1, 80, 300]

                        # Set sparsity
                        if(self._ran_sparsity and reviews_ctr == self._reviews_be_chosen):
                            outputs = torch.randn((1, 80, 300) , device=self.device)
                            
                        if(reviews_ctr == 0):
                            interInput = outputs
                            interInput_asin = this_asins
                        else:
                            interInput = torch.cat((interInput, outputs) , 0) 
                            interInput_asin = torch.cat((interInput_asin, this_asins) , 0)


                        if(concat_rating):
                            this_rating = self.testing_review_rating[reviews_ctr][batch_ctr]

                            _encode_rating = self._rating_to_onehot(this_rating)
                            _encode_rating = torch.tensor(_encode_rating).to(self.device)
                            _encode_rating = _encode_rating.unsqueeze(0)
                            pass
                        else:
                            inter_intput_rating =None           


                        if(reviews_ctr == 0):
                            # interInput = base_net_outputs
                            if(concat_rating):
                                inter_intput_rating = _encode_rating
                        else:
                            if(concat_rating):
                                inter_intput_rating = torch.cat(
                                    (inter_intput_rating, _encode_rating) , 0
                                    )

                    # # Writing Intra-attention weight to .html file
                    # if(isWriteAttn):
                        
                    #     """Only considerate user attention"""
                    #     current_candidates = current_reviewerIDs
                        
                    #     for index_ , candidateObj_ in enumerate(current_candidates):

                    #         intra_attn_wts = intra_attn_score[:,index_].squeeze(1).tolist()
                    #         word_indexes = input_variable[:,index_].tolist()
                            
                    #         sentence, weights = AttnVisualize.wdIndex2sentences(word_indexes, self.voc.index2word, intra_attn_wts)
                    #         AttnVisualize.createHTML(
                    #             sentence, 
                    #             weights, 
                    #             reviews_ctr, 
                    #             fname='{}@{}'.format( userObj.index2reviewerID[candidateObj_.item()], reviews_ctr)
                    #             )                           
                                
                with torch.no_grad():
                    outputs, intra_hidden, inter_attn_score  = InterGRU(
                        interInput, 
                        interInput_asin, 
                        current_asins, 
                        current_reviewerIDs,
                        review_rating = inter_intput_rating
                        )
                    outputs = outputs.squeeze(1)


                """correspond"""
                # Forward pass through HANN
                for reviews_ctr in range(4): # iter. through reviews

                    current_batch = correspond_batches[reviews_ctr][batch_ctr]            
                    input_variable, lengths, ratings = current_batch
                    input_variable = input_variable.to(self.device)
                    lengths = lengths.to(self.device)

                    correspond_current_asins = torch.tensor(validate_asins[idx][batch_ctr]).to(self.device)
                    correspond_current_reviewerIDs = torch.tensor(validate_reviewerIDs[idx][batch_ctr]).to(self.device)
                    

                    with torch.no_grad():
                        correspond_net_outputs, intra_hidden, intra_attn_score = correspond_intra_GRU[reviews_ctr](input_variable, lengths, 
                            correspond_current_asins, correspond_current_reviewerIDs)
                        correspond_net_outputs = correspond_net_outputs.unsqueeze(0)

                        if(reviews_ctr == 0):
                            interInput = correspond_net_outputs
                            interInput_asin = this_asins
                        else:
                            interInput = torch.cat((interInput, correspond_net_outputs) , 0) 
                            interInput_asin = torch.cat((interInput_asin, this_asins) , 0) 

                        if(concat_rating):
                            this_rating = self.testing_correspond_review_rating[reviews_ctr][batch_ctr]

                            _encode_rating = self._rating_to_onehot(this_rating)
                            _encode_rating = torch.tensor(_encode_rating).to(self.device)
                            _encode_rating = _encode_rating.unsqueeze(0)
                            pass
                        else:
                            inter_intput_rating =None           


                        if(reviews_ctr == 0):
                            # interInput = base_net_outputs
                            if(concat_rating):
                                inter_intput_rating = _encode_rating
                        else:
                            if(concat_rating):
                                inter_intput_rating = torch.cat(
                                    (inter_intput_rating, _encode_rating) , 0
                                    ) 

                    # Writing Intra-attention weight to .html file
                    if(isWriteAttn):
                        
                        """Only considerate user attention"""
                        current_candidates = correspond_current_reviewerIDs
                        
                        for index_ , candidateObj_ in enumerate(current_candidates):

                            intra_attn_wts = intra_attn_score[:,index_].squeeze(1).tolist()
                            word_indexes = input_variable[:,index_].tolist()
                            sentence, weights = AttnVisualize.wdIndex2sentences(word_indexes, self.voc.index2word, intra_attn_wts)
                            AttnVisualize.createHTML(
                                sentence, 
                                weights, 
                                reviews_ctr, 
                                fname='{}@{}'.format( candidateObj.index2reviewerID[candidateObj_.item()], reviews_ctr)
                                )       

                with torch.no_grad():
                    correspond_net_outputs, intra_hidden, inter_attn_score  = correspond_inter_GRU(
                        interInput, 
                        interInput_asin, 
                        current_asins, 
                        current_reviewerIDs,
                        review_rating = inter_intput_rating
                        )
                    correspond_net_outputs = correspond_net_outputs.squeeze(1)


                with torch.no_grad():
                    finalOut = MFC(outputs, correspond_net_outputs, current_asins, current_reviewerIDs)
                    finalOut = finalOut.squeeze(1)

                
                opt_method = 'se'
                if(opt_method == 'se'):
                    # Caculate loss 
                    current_labels = torch.tensor(validate_batch_labels[idx][batch_ctr]).to(self.device)

                    err = (finalOut*(5-1)+1) - current_labels
                    loss = torch.mul(err, err)
                    loss = torch.mean(loss, dim=0)
                    loss = torch.sqrt(loss)
                    
                    # Calculate accuracy
                    _total = len(current_labels)
                    _count = 0
                    predict_rating = (finalOut*(5-1)+1).round()
                    for _key, _val in enumerate(current_labels):
                        if(predict_rating[_key] == current_labels[_key]):
                            _count += 1
                            pass

                        true_label.append(int(current_labels[_key]))
                        predict_label.append(int(predict_rating[_key]))

                        pass

                    _accuracy += float(_count/_total)                    

                elif(opt_method == 'nll'):
                    loss = 0
                    current_labels = torch.tensor(validate_batch_labels[idx][batch_ctr]).to(self.device)
                    for index_, label_ in enumerate(current_labels):
                        true_lb = int(label_.item())
                        tmp = finalOut[index_][true_lb-1]
                        # Calculate nll
                        _ = torch.log(
                            finalOut[index_][true_lb-1]
                            )

                        loss += _

                        predict_rating = torch.argmax(finalOut[index_])
                        true_label.append(int(current_labels[index_]))
                        predict_label.append(int(predict_rating.item())+1)


                        pass
                
                if(opt_method == 'se'):
                    group_loss += loss
                    pass
                elif(opt_method == 'nll'):
                    # Calculate nll
                    group_loss += (-1)*loss

                    # Calculate accuracy
                    _total = len(true_label)
                    _count = 0
                    for _key, _val in enumerate(true_label):
                        if(predict_label[_key] == true_label[_key]):
                            _count += 1

                    _accuracy += float(_count/_total)      

                    pass                 
                    
        

        num_of_iter = len(training_batches[0])*len(validate_batch_labels)


        # Calculate confusion matrix
        cnf_matrix = confusion_matrix(true_label, predict_label)

        if(opt_method == 'nll'):
            NLL = group_loss/num_of_iter
            Accuracy = _accuracy/num_of_iter
            return NLL, Accuracy, cnf_matrix

        RMSE = group_loss/num_of_iter
        Accuracy = _accuracy/num_of_iter


        return RMSE, Accuracy, cnf_matrix

    def _rating_to_onehot(self, rating, rating_dim=5):
        # Initial onehot table
        onehot_table = [0 for _ in range(rating_dim)]
        rating = [int(val-1) for val in rating]
        
        _encode_rating = list()
        for val in rating:
            current_onehot = onehot_table.copy()    # copy from init.
            current_onehot[val] = 1.0               # set rating as onehot
            _encode_rating.append(current_onehot)

        return _encode_rating


    def plot_confusion_matrix(self, cm, classes,
                            normalize = False,
                            title = 'Confusion matrix',
                            cmap = plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            # print("Normalized confusion matrix")
        else:
            # print('Confusion matrix, without normalization')
            pass

        # print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()