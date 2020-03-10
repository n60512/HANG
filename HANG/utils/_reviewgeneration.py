import torch
import torch.nn as nn
from torch import optim

import tqdm
import random
from utils.model import IntraReviewGRU, HANN, DecoderGRU
from utils.setup import train_test_setup
from visualization.attention_visualization import Visualization

class ReviewGeneration(train_test_setup):
    def __init__(self, device, net_type, save_dir, voc, prerocess, 
        training_epoch=100, latent_k=32, hidden_size=300, clip=50,
        num_of_reviews = 5, 
        intra_method='dualFC', inter_method='dualFC', 
        learning_rate=0.00001, dropout=0):
        
        super(ReviewGeneration, self).__init__(device, net_type, save_dir, voc, prerocess, training_epoch, latent_k, hidden_size, clip, num_of_reviews, intra_method, inter_method, learning_rate, dropout)

        # Default word tokens
        self.PAD_token = 0  # Used for padding short sentences
        self.SOS_token = 1  # Start-of-sentence token
        self.EOS_token = 2  # End-of-sentence token   
        self.batch_size = 40     
        self.teacher_forcing_ratio = 1.0
        
        pass    

    def set_label_sentences(self, label_sentences):
        """Setup label_sentences for GRM"""
        self.label_sentences = label_sentences
        pass

    def set_decoder_learning_ratio(self, decoder_learning_ratio):
        self.decoder_learning_ratio = decoder_learning_ratio
        pass

    def _load_pretrain_hann(self, pretrain_model_path, _ep):
        """Using pretrain hann to initial GRM"""
        
        # Initialize IntraGRU models
        IntraGRU = list()

        for idx in range(self.num_of_reviews):
            intra_model_path = '{}/IntraGRU_idx{}_epoch{}'.format(pretrain_model_path, idx, _ep)
            model = torch.load(intra_model_path)
            IntraGRU.append(model)

        # Loading InterGRU
        inter_model_path = '{}/InterGRU_epoch{}'.format(pretrain_model_path, _ep)
        InterGRU = torch.load(inter_model_path)

        return IntraGRU, InterGRU

    def _train_iteration_grm(self, IntraGRU, InterGRU, DecoderModel, IntraGRU_optimizer, InterGRU_optimizer, DecoderModel_optimizer,
        training_batches, external_memorys, candidate_items, candidate_users, training_batch_labels, label_sentences,
        isCatItemVec=False):
        """ Training each iteraction"""

        # Initialize this epoch loss
        hann_epoch_loss = 0
        decoder_epoch_loss = 0

        for batch_ctr in tqdm.tqdm(range(len(training_batches[0]))): # amount of batches
            # Run multiple label for training 
            for idx in range(len(training_batch_labels)):

                # If turning HANN
                if(self.tuning_hann):
                    InterGRU_optimizer.zero_grad()
                    for reviews_ctr in range(len(training_batches)): # iter. through reviews
                        IntraGRU_optimizer[reviews_ctr].zero_grad()

                # Forward pass through HANN
                for reviews_ctr in range(len(training_batches)): # iter. through reviews

                    current_batch = training_batches[reviews_ctr][batch_ctr]            
                    input_variable, lengths, ratings = current_batch
                    input_variable = input_variable.to(self.device)
                    lengths = lengths.to(self.device)

                    current_asins = torch.tensor(candidate_items[idx][batch_ctr]).to(self.device)
                    current_reviewerIDs = torch.tensor(candidate_users[idx][batch_ctr]).to(self.device)

                    outputs, intra_hidden, intra_attn_score = IntraGRU[reviews_ctr](input_variable, lengths, 
                        current_asins, current_reviewerIDs)
                    
                    outputs = outputs.unsqueeze(0)

                    if(isCatItemVec):
                        # Concat. asin feature
                        this_asins = external_memorys[reviews_ctr][batch_ctr]
                        this_asins = torch.tensor([val for val in this_asins]).to(self.device)
                        this_asins = this_asins.unsqueeze(0)
                    else:
                        interInput_asin = None

                    if(reviews_ctr == 0):
                        interInput = outputs
                        if(isCatItemVec):
                            interInput_asin = this_asins
                    else:
                        interInput = torch.cat((interInput, outputs) , 0) 
                        if(isCatItemVec):
                            interInput_asin = torch.cat((interInput_asin, this_asins) , 0) 


                outputs, inter_hidden, inter_attn_score  = InterGRU(interInput, interInput_asin, current_asins, current_reviewerIDs)
                outputs = outputs.squeeze(1)


                # Caculate loss 
                current_labels = torch.tensor(training_batch_labels[idx][batch_ctr]).to(self.device)
                err = (outputs*(5-1)+1) - current_labels

                hann_loss = torch.mul(err, err)
                hann_loss = torch.mean(hann_loss, dim=0)

                # HANN loss of this epoch
                hann_epoch_loss += hann_loss
                


                """
                Runing Decoder
                """
                # Ground true sentences
                target_batch = label_sentences[0][batch_ctr]
                target_variable, target_len, _ = target_batch 
                max_target_len = max(target_len)
                target_variable = target_variable.to(self.device)  

                # Create initial decoder input (start with SOS tokens for each sentence)
                decoder_input = torch.LongTensor([[self.SOS_token for _ in range(self.batch_size)]])
                decoder_input = decoder_input.to(self.device)   


                # Set initial decoder hidden state to the inter_hidden's final hidden state
                criterion = nn.NLLLoss()
                decoder_loss = 0
                decoder_hidden = inter_hidden

                # Determine if we are using teacher forcing this iteration
                use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

                # Forward batch of sequences through decoder one time step at a time
                if use_teacher_forcing:
                    for t in range(max_target_len):
                        decoder_output, decoder_hidden = DecoderModel(
                            decoder_input, decoder_hidden
                        )
                        # Teacher forcing: next input is current target
                        decoder_input = target_variable[t].view(1, -1)  # get the row(word) of sentences

                        # Calculate and accumulate loss
                        nll_loss = criterion(decoder_output, target_variable[t])
                        decoder_loss += nll_loss
                else:
                    for t in range(max_target_len):
                        decoder_output, decoder_hidden = DecoderModel(
                            decoder_input, decoder_hidden
                        )
                        # No teacher forcing: next input is decoder's own current output
                        _, topi = decoder_output.topk(1)

                        decoder_input = torch.LongTensor([[topi[i][0] for i in range(self.batch_size)]])
                        decoder_input = decoder_input.to(device)

                        # Calculate and accumulate loss
                        nll_loss = criterion(decoder_output, target_variable[t])
                        decoder_loss += nll_loss


                loss = hann_loss + decoder_loss

                # Perform backpropatation
                loss.backward()

                # Clip gradients: gradients are modified in place
                for reviews_ctr in range(len(training_batches)):            
                    _ = nn.utils.clip_grad_norm_(IntraGRU[reviews_ctr].parameters(), self.clip)
                _ = nn.utils.clip_grad_norm_(InterGRU.parameters(), self.clip)


                # If turning HANN
                if(self.tuning_hann):
                    # Adjust `HANN` model weights
                    for reviews_ctr in range(len(training_batches)):
                        IntraGRU_optimizer[reviews_ctr].step()
                    InterGRU_optimizer.step()

                # Adjust Decoder model weights
                DecoderModel_optimizer.step()

                # decoder loss of this epoch
                decoder_epoch_loss += decoder_loss.item()/float(max_target_len)

        return hann_epoch_loss, decoder_epoch_loss

    def set_tune_option(self, user_pretrain_hann=False, tuning_hann=True):
        self.user_pretrain_hann = user_pretrain_hann
        self.tuning_hann = tuning_hann
        pass

    def train_grm(self, select_table, isStoreModel=False, isStoreCheckPts=False, WriteTrainLoss=False, store_every = 2, 
            use_pretrain_item= False, isCatItemVec= True, pretrain_wordVec=None):
        
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


        if(self.user_pretrain_hann):
            pretrain_model_path = ''
            _ep =''
            IntraGRU, InterGRU = self._load_pretrain_hann(pretrain_model_path, _ep)

            # Use appropriate device
            InterGRU = InterGRU.to(self.device)
            for idx in range(self.num_of_reviews):    
                IntraGRU[idx] = IntraGRU[idx].to(self.device)            
        
        else:

            # Append GRU model asc
            for idx in range(self.num_of_reviews):    
                IntraGRU.append(IntraReviewGRU(self.hidden_size, embedding, asin_embedding, reviewerID_embedding,  
                    latentK = self.latent_k, method=self.intra_method))
                # Use appropriate device
                IntraGRU[idx] = IntraGRU[idx].to(self.device)
                IntraGRU[idx].train()

            # Initialize InterGRU models
            InterGRU = HANN(self.hidden_size, embedding, asin_embedding, reviewerID_embedding,
                    n_layers=1, dropout=self.dropout, latentK = self.latent_k, isCatItemVec=isCatItemVec , 
                    netType=self.net_type, method=self.inter_method)

            # Use appropriate device
            InterGRU = InterGRU.to(self.device)
            InterGRU.train()

        if(self.tuning_hann):
            IntraGRU_optimizer = list()
            # Initialize IntraGRU optimizers groups
            intra_scheduler = list()

            for idx in range(self.num_of_reviews):    
                # Initialize optimizers
                IntraGRU_optimizer.append(optim.AdamW(IntraGRU[idx].parameters(), 
                        lr=self.learning_rate, weight_decay=0.001)
                    )            
                # Assuming optimizer has two groups.
                intra_scheduler.append(optim.lr_scheduler.StepLR(IntraGRU_optimizer[idx], 
                    step_size=20, gamma=0.3))

            # Initialize IntraGRU optimizers    
            InterGRU_optimizer = optim.AdamW(InterGRU.parameters(), 
                    lr=self.learning_rate, weight_decay=0.001)
            # Assuming optimizer has two groups.
            inter_scheduler = optim.lr_scheduler.StepLR(InterGRU_optimizer, 
                step_size=10, gamma=0.3)
        else:
            IntraGRU_optimizer = None
            InterGRU_optimizer = None

        # Initialize DecoderGRU models and optimizers
        DecoderModel = DecoderGRU(embedding, self.hidden_size, self.voc.num_words, n_layers=1, dropout=self.dropout)
        # Use appropriate device
        DecoderModel = DecoderModel.to(self.device)
        DecoderModel.train()
        # Initialize DecoderGRU optimizers    
        DecoderModel_optimizer = optim.AdamW(DecoderModel.parameters(), 
                lr=self.learning_rate * self.decoder_learning_ratio, 
                weight_decay=0.001)    

        print('Models built and ready to go!')        

        for Epoch in range(self.training_epoch):
            # Run a training iteration with batch
            hann_group_loss, decoder_group_loss = self._train_iteration_grm(IntraGRU, InterGRU, DecoderModel, IntraGRU_optimizer, InterGRU_optimizer, DecoderModel_optimizer,
                self.training_batches, self.external_memorys, self.candidate_items, self.candidate_users, self.training_batch_labels, self.label_sentences,
                isCatItemVec=isCatItemVec)

            # IF tuning HANN
            if(self.tuning_hann):
                inter_scheduler.step()
                for idx in range(self.num_of_reviews):
                    intra_scheduler[idx].step()
                    
            num_of_iter = len(self.training_batches[0])*len(self.training_batch_labels)
        
            hann_loss_average = hann_group_loss/num_of_iter
            decoder_loss_average = decoder_group_loss/num_of_iter

            print('Epoch:{}\tHANN(SE):{}\tNNL:{}\t'.format(Epoch, hann_loss_average, decoder_loss_average))

            if(Epoch % store_every == 0 and isStoreModel):
                torch.save(InterGRU, R'{}/Model/InterGRU_epoch{}'.format(self.save_dir, Epoch))
                torch.save(DecoderModel, R'{}/Model/DecoderModel_epoch{}'.format(self.save_dir, Epoch))
                for idx__, IntraGRU__ in enumerate(IntraGRU):
                    torch.save(IntraGRU__, R'{}/Model/IntraGRU_idx{}_epoch{}'.format(self.save_dir, idx__, Epoch))
    
            if WriteTrainLoss:
                with open(R'{}/Loss/TrainingLoss.txt'.format(self.save_dir),'a') as file:
                    file.write('Epoch:{}\tHANN(SE):{}\tNNL:{}\n'.format(Epoch, hann_loss_average, decoder_loss_average))

        pass