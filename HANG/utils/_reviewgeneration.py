import torch
import torch.nn as nn
from torch import optim
import nltk
from nltk.corpus import stopwords

import tqdm
import random
from rouge import Rouge
from utils.model import IntraReviewGRU, HANN, DecoderGRU, nrt_rating_predictor, nrt_decoder, HANN_new
from utils.setup import train_test_setup
from visualization.attention_visualization import Visualization
from torchnlp.metrics import get_moses_multi_bleu

class ReviewGeneration(train_test_setup):
    def __init__(self, device, net_type, save_dir, voc, prerocess, 
        training_epoch=100, latent_k=32, batch_size=40, hidden_size=300, clip=50,
        num_of_reviews = 5,
        intra_method='dualFC', inter_method='dualFC', 
        learning_rate=0.00001, dropout=0,
        setence_max_len=50):
        
        super(ReviewGeneration, self).__init__(device, net_type, save_dir, voc, prerocess, training_epoch, latent_k, batch_size, hidden_size, clip, num_of_reviews, intra_method, inter_method, learning_rate, dropout, setence_max_len)

        # Default word tokens
        self.PAD_token = 0  # Used for padding short sentences
        self.SOS_token = 1  # Start-of-sentence token
        self.EOS_token = 2  # End-of-sentence token   

        # self.teacher_forcing_ratio = 0.8
        self.teacher_forcing_ratio = 1.0
        pass    

    def set_object(self, userObj, itemObj):
        self.userObj = userObj
        self.itemObj = itemObj
        pass

    def set_training_review_rating(self, training_review_rating):
        self.training_review_rating = training_review_rating
        pass

    def set_testing_review_rating(self, testing_review_rating):
        self.testing_review_rating = testing_review_rating
        pass

    def set_label_sentences(self, label_sentences):
        """Setup label_sentences for GRM"""
        self.label_sentences = label_sentences
        pass

    def set_decoder_learning_ratio(self, decoder_learning_ratio):
        self.decoder_learning_ratio = decoder_learning_ratio
        pass

    def set_testing_batches(self, testing_batches, testing_external_memorys, testing_batch_labels, testing_asins, testing_reviewerIDs, testing_label_sentences):
        self.testing_batches = testing_batches
        self.testing_external_memorys = testing_external_memorys
        self.testing_batch_labels = testing_batch_labels
        self.testing_asins = testing_asins
        self.testing_reviewerIDs = testing_reviewerIDs
        self.testing_label_sentences = testing_label_sentences
        pass

    def set_tune_option(self, use_pretrain_item_net=False, tuning_item_net=True):
        self.use_pretrain_item_net = use_pretrain_item_net
        self.tuning_item_net = tuning_item_net
        pass    

    def _load_pretrain_item_net(self, pretrain_model_path, _ep):
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

    def _train_iteration_grm(self, IntraGRU, InterGRU, DecoderModel, 
        IntraGRU_optimizer, InterGRU_optimizer, DecoderModel_optimizer,
        training_batches, external_memorys, candidate_items, candidate_users, 
        training_batch_labels, label_sentences,
        isCatItemVec=False, concat_rating=False, 
        _use_coverage=False, fix_rating=False):
        """ Training each iteraction"""

        # Initialize this epoch loss
        hann_epoch_loss = 0
        decoder_epoch_loss = 0

        for batch_ctr in tqdm.tqdm(range(len(training_batches[0]))): # amount of batches
            # Run multiple label for training 
            for idx in range(len(training_batch_labels)):
                
                
                # If turning HANN
                if(self.tuning_item_net):
                    InterGRU_optimizer.zero_grad()
                    for reviews_ctr in range(len(training_batches)): # iter. through reviews
                        IntraGRU_optimizer[reviews_ctr].zero_grad()

                DecoderModel_optimizer.zero_grad()
                    

                # Forward pass through intra gru
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


                    """ Cat item vector """
                    if(isCatItemVec):
                        # Concat. asin feature
                        this_asins = external_memorys[reviews_ctr][batch_ctr]
                        this_asins = torch.tensor([val for val in this_asins]).to(self.device)
                        this_asins = this_asins.unsqueeze(0)
                    else:
                        interInput_asin = None

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
                        if(isCatItemVec):
                            interInput_asin = this_asins
                    else:
                        interInput = torch.cat((interInput, outputs) , 0) 
                        if(isCatItemVec):
                            interInput_asin = torch.cat((interInput_asin, this_asins) , 0) 

                        if(concat_rating):
                            inter_intput_rating = torch.cat(
                                (inter_intput_rating, _encode_rating) , 0
                                )                             

                # Forward pass through inter gru
                outputs, inter_hidden, inter_attn_score, context_vector  = InterGRU(
                    interInput, 
                    interInput_asin, 
                    current_asins, 
                    current_reviewerIDs,
                    review_rating = inter_intput_rating
                    )
                outputs = outputs.squeeze(1)

                """ 
                Caculate Square Loss 
                """
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
                coverage_loss = 0

                """NEXT STEP : CONCAT. RATING"""
                # dec_merge_rating = True
                if(True):
                    _encode_rating = list()
                    _encode_rating = self._rating_to_onehot(current_labels)
                    _encode_rating = torch.tensor(_encode_rating).to(self.device)
                    _encode_rating = _encode_rating.unsqueeze(0)
             
                decoder_hidden = inter_hidden
                _enable_attention = True

                # Determine if we are using teacher forcing this iteration
                use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

                # Forward batch of sequences through decoder one time step at a time
                if use_teacher_forcing:
                    for t in range(max_target_len):

                        if(t == 0 and _use_coverage):
                            # Set up initial coverage probability
                            # initial_coverage_prob = torch.randn(1, 80, 44504)
                            # initial_coverage_prob = torch.zeros(1, 80, 44519)

                            initial_coverage_prob = torch.zeros(1, self.batch_size, self.voc.num_words)


                            initial_coverage_prob = initial_coverage_prob.to(self.device)
                            DecoderModel.set_coverage_prob(initial_coverage_prob, _use_coverage)
                        
                        # Forward pass through decoder
                        decoder_output, decoder_hidden, decoder_attn_weight = DecoderModel(
                            decoder_input, 
                            decoder_hidden, 
                            context_vector,
                            _encode_rating = _encode_rating,
                            _user_emb = current_reviewerIDs,
                            _item_emb = current_asins,
                            _enable_attention = _enable_attention
                        )
                        # Teacher forcing: next input is current target
                        decoder_input = target_variable[t].view(1, -1)  # get the row(word) of sentences
                        
                        """
                        coverage
                        """
                        if(_use_coverage):
                            _softmax_output = DecoderModel.get_softmax_output()
                            _current_prob = _softmax_output.unsqueeze(0)

                            if(t==0):
                                _previous_prob_sum = _current_prob
                            else:
                                # sum up previous probability
                                _previous_prob_sum = _previous_prob_sum + _current_prob
                                DecoderModel.set_coverage_prob(_previous_prob_sum, _use_coverage)

                            tmp = torch.cat((_previous_prob_sum, _current_prob), dim = 0)
                            # extract min values
                            _coverage_mechanism_ = torch.min(tmp, dim = 0).values

                            _coverage_mechanism_sum = torch.sum(_coverage_mechanism_, dim=1)
                            coverage_loss += torch.sum(_coverage_mechanism_sum, dim=0)
                            pass

                        # Calculate and accumulate loss
                        nll_loss = criterion(decoder_output, target_variable[t])
                        decoder_loss += nll_loss
                else:
                    for t in range(max_target_len):

                        if(t == 0 and _use_coverage):
                            # Set up initial coverage probability
                            # initial_coverage_prob = torch.randn(1, 80, 44504)
                            # initial_coverage_prob = torch.zeros(1, 80, 44519)

                            initial_coverage_prob = torch.zeros(1, self.batch_size, self.voc.num_words)


                            initial_coverage_prob = initial_coverage_prob.to(self.device)
                            DecoderModel.set_coverage_prob(initial_coverage_prob, _use_coverage)
                        
                        # Forward pass through decoder
                        decoder_output, decoder_hidden, decoder_attn_weight = DecoderModel(
                            decoder_input, 
                            decoder_hidden, 
                            context_vector,
                            _encode_rating = _encode_rating,
                            _user_emb = current_reviewerIDs,
                            _item_emb = current_asins,
                            _enable_attention = _enable_attention
                        )
                        # No teacher forcing: next input is decoder's own current output
                        _, topi = decoder_output.topk(1)

                        decoder_input = torch.LongTensor([[topi[i][0] for i in range(self.batch_size)]])
                        decoder_input = decoder_input.to(self.device)
                        
                        """
                        coverage
                        """
                        if(_use_coverage):
                            _softmax_output = DecoderModel.get_softmax_output()
                            _current_prob = _softmax_output.unsqueeze(0)

                            if(t==0):
                                _previous_prob_sum = _current_prob
                            else:
                                # sum up previous probability
                                _previous_prob_sum = _previous_prob_sum + _current_prob
                                DecoderModel.set_coverage_prob(_previous_prob_sum, _use_coverage)

                            tmp = torch.cat((_previous_prob_sum, _current_prob), dim = 0)
                            # extract min values
                            _coverage_mechanism_ = torch.min(tmp, dim = 0).values

                            _coverage_mechanism_sum = torch.sum(_coverage_mechanism_, dim=1)
                            coverage_loss += torch.sum(_coverage_mechanism_sum, dim=0)
                            pass

                        # Calculate and accumulate loss
                        nll_loss = criterion(decoder_output, target_variable[t])
                        decoder_loss += nll_loss


                
                # freeze.parameters
                if(fix_rating):
                    for p in InterGRU.parameters():
                        p.requires_grad = False
                    
                    for _model in IntraGRU:                        
                        for p in _model.parameters():
                            p.requires_grad = False


                loss = hann_loss + 0.7 * (decoder_loss/max_target_len)

                # if(_use_coverage):
                #     loss += (0)*coverage_loss

                # Perform backpropatation
                loss.backward()

                # If turning HANN
                if(self.tuning_item_net):
                    # Adjust `HANN` model weights
                    for reviews_ctr in range(len(training_batches)):
                        IntraGRU_optimizer[reviews_ctr].step()
                    InterGRU_optimizer.step()

                    # Clip gradients: gradients are modified in place
                    for reviews_ctr in range(len(training_batches)):            
                        _ = nn.utils.clip_grad_norm_(IntraGRU[reviews_ctr].parameters(), self.clip)
                    _ = nn.utils.clip_grad_norm_(InterGRU.parameters(), self.clip)                    

                # Adjust Decoder model weights
                DecoderModel_optimizer.step()

                # decoder loss of this epoch
                decoder_epoch_loss += decoder_loss.item()/float(max_target_len)

        return hann_epoch_loss, decoder_epoch_loss

    def train_grm(self, select_table, isStoreModel = False, isStoreCheckPts = False, ep_to_store = 0, 
            WriteTrainLoss=False, store_every = 2, use_pretrain_item = False, 
            isCatItemVec = False, concat_rating = False, pretrain_wordVec = None, 
            _use_coverage = False):

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

        # Using pretain item net for multi-tasking
        if(self.use_pretrain_item_net):
            pretrain_model_path = 'HANG/data/pretrain_itembase_model'
            # _ep ='2'  # 0317 version
            # _ep ='36'  # 0427 version
            # _ep ='62'  # 0520 add rating version
            # _ep='24'
            # _ep ='20'  # 0520 add rating & content version
            # _ep ='36'  # 0521 add rating version
            # _ep ='26'  # 0521 add rating version
            _ep ='52'  # 0521 add rating version
            # _ep ='11'  # 0701 full interaction pre-train
            _ep ='10'  # 0705 _all_interaction6_item.rgm.full.turn1.8.1 PRETRAIN
            IntraGRU, InterGRU = self._load_pretrain_item_net(pretrain_model_path, _ep)

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
            InterGRU = HANN_new(self.hidden_size, embedding, asin_embedding, reviewerID_embedding,
                    n_layers=1, dropout=self.dropout, latentK = self.latent_k, 
                    isCatItemVec=isCatItemVec , concat_rating= concat_rating,
                    netType=self.net_type, method=self.inter_method
                    )

            # Use appropriate device
            InterGRU = InterGRU.to(self.device)
            InterGRU.train()

        # Wheather tune item net
        if(self.tuning_item_net):
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
        DecoderModel = DecoderGRU(
            embedding, 
            self.hidden_size, 
            self.voc.num_words, 
            n_layers=1, 
            dropout=self.dropout
            )

        DecoderModel.set_user_embedding(reviewerID_embedding)
        DecoderModel.set_item_embedding(asin_embedding)

        # Use appropriate device
        DecoderModel = DecoderModel.to(self.device)
        DecoderModel.train()
        # Initialize DecoderGRU optimizers    
        DecoderModel_optimizer = optim.AdamW(DecoderModel.parameters(), 
                lr=self.learning_rate * self.decoder_learning_ratio, 
                weight_decay=0.001
                )

        print('Models built and ready to go!')

        RMSE = 99
        _flag = True
        fix_rating = False

        # Training model
        for Epoch in range(self.training_epoch):

            if(_flag):
                if RMSE < 1.076 and Epoch>2:
                    _flag = False
                    fix_rating = True

            print("ep:{}, _flag:{}".format(Epoch, _flag))

            # if Epoch > 20:
            #     self.teacher_forcing_ratio = 0.6
            # elif Epoch > 10:
            #     self.teacher_forcing_ratio = 0.5

            # Run a training iteration with batch
            hann_group_loss, decoder_group_loss = self._train_iteration_grm(
                IntraGRU, InterGRU, DecoderModel, 
                IntraGRU_optimizer, InterGRU_optimizer, DecoderModel_optimizer,
                self.training_batches, self.external_memorys, self.candidate_items, 
                self.candidate_users, self.training_batch_labels, self.label_sentences,
                isCatItemVec=isCatItemVec, concat_rating=concat_rating,
                _use_coverage=_use_coverage,
                fix_rating=fix_rating 
                )

            # Wheather tune item net
            if(self.tuning_item_net):
                inter_scheduler.step()
                for idx in range(self.num_of_reviews):
                    intra_scheduler[idx].step()
                    
            num_of_iter = len(self.training_batches[0])*len(self.training_batch_labels)
        
            hann_loss_average = hann_group_loss/num_of_iter
            decoder_loss_average = decoder_group_loss/num_of_iter

            print('Epoch:{}\tItemNet(SE):{}\tNNL:{}\t'.format(Epoch, hann_loss_average, decoder_loss_average))

            if(Epoch % store_every == 0 and isStoreModel and Epoch >= ep_to_store):
                torch.save(InterGRU, R'{}/Model/InterGRU_epoch{}'.format(self.save_dir, Epoch))
                torch.save(DecoderModel, R'{}/Model/DecoderModel_epoch{}'.format(self.save_dir, Epoch))
                for idx__, IntraGRU__ in enumerate(IntraGRU):
                    torch.save(IntraGRU__, R'{}/Model/IntraGRU_idx{}_epoch{}'.format(self.save_dir, idx__, Epoch))
    
            if WriteTrainLoss:
                with open(R'{}/Loss/TrainingLoss.txt'.format(self.save_dir),'a') as file:
                    file.write('Epoch:{}\tItemNet(SE):{}\tNNL:{}\n'.format(Epoch, hann_loss_average, decoder_loss_average))

            """
            Evaluating BLEU
            """
            # evaluating
            RMSE, _nllloss, batch_bleu_score, average_rouge_score = self.evaluate_generation(
                IntraGRU, 
                InterGRU, 
                DecoderModel, 
                Epoch,
                isCatItemVec=isCatItemVec, 
                concat_rating=concat_rating,
                write_insert_sql=True,
                write_origin=True,
                _use_coverage = _use_coverage,
                _write_mode = 'evaluate'
                )

            print('Epoch:{}\tMSE:{}\tNNL:{}\t'.format(Epoch, RMSE, _nllloss))
            with open(R'{}/Loss/ValidationLoss.txt'.format(self.save_dir),'a') as file:
                file.write('Epoch:{}\tRMSE:{}\tNNL:{}\n'.format(Epoch, RMSE, _nllloss))   

            for num, val in enumerate(batch_bleu_score):
                with open('{}/Bleu/Validation/blue{}.score.txt'.format(self.save_dir, (num+1)),'a') as file:
                    file.write('BLEU SCORE {}.ep.{}: {}\n'.format((num+1), Epoch, val))
                print('BLEU SCORE {}: {}'.format((num+1), val))

            with open('{}/Bleu/Validation/rouge.score.txt'.format(self.save_dir), 'a') as file:
                file.write('=============================\nEpoch:{}\n'.format(Epoch))
                for _rouge_method, _metrics in average_rouge_score.items():
                    for _key, _val in _metrics.items():
                        file.write('{}. {}: {}\n'.format(_rouge_method, _key, _val))
                        print('{}. {}: {}'.format(_rouge_method, _key, _val))
            
        pass

    def evaluate_generation(self, 
        IntraGRU, InterGRU, DecoderModel, Epoch, 
        isCatItemVec=False, concat_rating=False,
        isWriteAttn=False,
        write_origin=False,
        write_insert_sql=False,
        _use_coverage=False,
        _write_mode = 'evaluate',
        visulize_attn_epoch = 0
        ):
        
        EngStopWords = set(stopwords.words('english'))

        group_loss = 0
        decoder_epoch_loss = 0
        AttnVisualize = Visualization(self.save_dir, visulize_attn_epoch, self.num_of_reviews)

        batch_bleu_score_1 = 0
        batch_bleu_score_2 = 0
        batch_bleu_score_3 = 0
        batch_bleu_score_4 = 0

        rouge = Rouge()

        average_rouge_score = {
            'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0},
            'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
            'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}
            }        

        for batch_ctr in tqdm.tqdm(range(len(self.testing_batches[0]))): #how many batches
            for idx in range(len(self.testing_batch_labels)):
                for reviews_ctr in range(len(self.testing_batches)): # iter. through reviews
                    
                    current_batch = self.testing_batches[reviews_ctr][batch_ctr]
                    
                    input_variable, lengths, ratings = current_batch
                    input_variable = input_variable.to(self.device)
                    lengths = lengths.to(self.device)

                    current_asins = torch.tensor(self.testing_asins[idx][batch_ctr]).to(self.device)
                    current_reviewerIDs = torch.tensor(self.testing_reviewerIDs[idx][batch_ctr]).to(self.device)
            
                    # Concat. external_memorys feature
                    this_reviewerIDs = self.testing_external_memorys[reviews_ctr][batch_ctr]
                    this_reviewerIDs = torch.tensor([val for val in this_reviewerIDs]).to(self.device)
                    this_reviewerIDs = this_reviewerIDs.unsqueeze(0)

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
                            interInput_reviewerIDs = this_reviewerIDs
                            if(concat_rating):
                                inter_intput_rating = _encode_rating                                 
                        else:
                            interInput = torch.cat((interInput, outputs) , 0) 
                            interInput_reviewerIDs = torch.cat((interInput_reviewerIDs, this_reviewerIDs) , 0) 

                            if(concat_rating):
                                inter_intput_rating = torch.cat(
                                    (inter_intput_rating, _encode_rating) , 0
                                    )                             

                    # Writing Intra-attention weight to .html file
                    if(_write_mode == 'attention'):

                        for index_ , candidateObj_ in enumerate(current_asins):

                            intra_attn_wts = intra_attn_score[:,index_].squeeze(1).tolist()
                            word_indexes = input_variable[:,index_].tolist()
                            sentence, weights = AttnVisualize.wdIndex2sentences(word_indexes, self.voc.index2word, intra_attn_wts)
                            
                            new_weights = [float(wts/sum(weights[0])) for wts in weights[0]]

                            for w_index, word in enumerate(sentence[0].split()):
                                if(word in EngStopWords):
                                    new_weights[w_index] = new_weights[w_index]*0.001
                                if(new_weights[w_index]<0.0001):
                                    new_weights[w_index] = 0

                            AttnVisualize.createHTML(
                                sentence, 
                                [new_weights], 
                                reviews_ctr,
                                # fname='{}@{}'.format( self.userObj.index2reviewerID[candidateObj_.item()], reviews_ctr)
                                fname='{}@{}'.format( self.itemObj.index2asin[candidateObj_.item()], reviews_ctr)
                                )

                with torch.no_grad():
                    outputs, inter_hidden, inter_attn_score, context_vector  = InterGRU(
                        interInput, 
                        interInput_reviewerIDs, 
                        current_asins, 
                        current_reviewerIDs,
                        review_rating = inter_intput_rating
                        )
                    outputs = outputs.squeeze(1)

                
                # Caculate Square loss of HANN 
                current_rating_labels = torch.tensor(self.testing_batch_labels[idx][batch_ctr]).to(self.device)
                
                predict_rating = (outputs*(5-1)+1)
                err = predict_rating - current_rating_labels
                loss = torch.mul(err, err)
                loss = torch.mean(loss, dim=0)
                loss = torch.sqrt(loss)
                group_loss += loss

                """
                Greedy Search Strategy Decoder
                """
                # Create initial decoder input (start with SOS tokens for each sentence)
                decoder_input = torch.LongTensor([[self.SOS_token for _ in range(self.batch_size)]])
                decoder_input = decoder_input.to(self.device)    


                dec_merge_rating = True
                if(dec_merge_rating):
                    current_labels = torch.tensor(self.testing_batch_labels[idx][batch_ctr]).to(self.device)
                    
                    # # all one test
                    # _all_one_point = [float(1.0) for _it in range(80)]
                    # current_labels = torch.FloatTensor(_all_one_point).to(self.device)

                    _encode_rating = list()
                    _encode_rating = self._rating_to_onehot(current_labels)
                    _encode_rating = torch.tensor(_encode_rating).to(self.device)
                    _encode_rating = _encode_rating.unsqueeze(0)


                # Set initial decoder hidden state to the inter_hidden's final hidden state
                decoder_hidden = inter_hidden

                criterion = nn.NLLLoss()
                decoder_loss = 0
                
                _enable_attention = True

                
                # Ground true sentences
                target_batch = self.testing_label_sentences[0][batch_ctr]
                target_variable, target_len, _ = target_batch   
                target_variable = target_variable.to(self.device)  

                # Generate max length
                max_target_len = self.setence_max_len

                # Initialize tensors to append decoded words to
                all_tokens = torch.zeros([0], device=self.device, dtype=torch.long)
                all_scores = torch.zeros([0], device=self.device)            
                
                """
                Greedy search
                """
                for t in range(max_target_len):

                    if(t == 0 and _use_coverage):
                        # Set up initial coverage probability
                        initial_coverage_prob = torch.zeros(1, self.batch_size, self.voc.num_words)
                        initial_coverage_prob = initial_coverage_prob.to(self.device)
                        DecoderModel.set_coverage_prob(initial_coverage_prob, _use_coverage)

                    decoder_output, decoder_hidden, decoder_attn_weight = DecoderModel(
                        decoder_input, 
                        decoder_hidden, 
                        context_vector,
                        _encode_rating = _encode_rating,
                        _user_emb = current_reviewerIDs,
                        _item_emb = current_asins,
                        _enable_attention = _enable_attention
                    )
                    # No teacher forcing: next input is decoder's own current output
                    decoder_scores_, topi = decoder_output.topk(1)

                    decoder_input = torch.LongTensor([[topi[i][0] for i in range(self.batch_size)]])
                    decoder_input = decoder_input.to(self.device)

                    ds, di = torch.max(decoder_output, dim=1)

                    # Record token and score
                    all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
                    all_scores = torch.cat((all_scores, torch.t(decoder_scores_)), dim=0)                


                    """
                    coverage
                    """
                    if(_use_coverage):
                        _softmax_output = DecoderModel.get_softmax_output()
                        _current_prob = _softmax_output.unsqueeze(0)

                        if(t==0):
                            _previous_prob_sum = _current_prob
                        else:
                            # sum up previous probability
                            _previous_prob_sum = _previous_prob_sum + _current_prob
                            DecoderModel.set_coverage_prob(_previous_prob_sum, _use_coverage)
                            pass
                        pass

                    # Calculate and accumulate loss
                    nll_loss = criterion(decoder_output, target_variable[t])
                    decoder_loss += nll_loss

                    pass

                # decoder loss of this epoch
                decoder_epoch_loss += decoder_loss.item()/float(max_target_len)

                """
                Decode user review from search result.
                """
                bleu_score_1 = 0
                bleu_score_2 = 0
                bleu_score_3 = 0
                bleu_score_4 = 0

                _rouge_score = {
                    'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                    'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                    'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}
                    }

                for index_ , user_ in enumerate(current_reviewerIDs):
                    
                    asin_ = current_asins[index_]

                    current_user_tokens = all_tokens[:,index_].tolist()
                    decoded_words = [self.voc.index2word[token] for token in current_user_tokens if token != 0]
                    predict_rating, current_rating_labels[index_].item()

                    try:
                        product_title = self.asin2title[
                            self.itemObj.index2asin[asin_.item()]
                        ]
                    except Exception as ex:
                        product_title = 'None'
                        pass


                    # Show user attention
                    if(_enable_attention):
                        inter_attn_score_ = inter_attn_score.squeeze(2).t()
                        this_user_attn = inter_attn_score_[index_]
                        this_user_attn = [str(val.item()) for val in this_user_attn]
                        attn_text = ' ,'.join(this_user_attn)                    
                    else:
                        attn_text = None

                    this_asin_input_reviewer = interInput_reviewerIDs.t()[index_]
                    input_reviewer = [self.userObj.index2reviewerID[val.item()] for val in this_asin_input_reviewer]                        

                    generate_text = str.format(
f"""
=========================
Userid & asin:{self.userObj.index2reviewerID[user_.item()]},{self.itemObj.index2asin[asin_.item()]}
title:{product_title}
pre. consumer:{' ,'.join(input_reviewer)}
Inter attn:{attn_text}
Predict:{predict_rating[index_].item()}
Rating:{current_rating_labels[index_].item()}
Generate: {' '.join(decoded_words)}
"""
                    )

                    if (write_origin):
                        current_user_sen = target_variable[:,index_].tolist()
                        origin_sen = [self.voc.index2word[token] for token in current_user_sen if token != 0]

                        generate_text = (
                            generate_text + 
                            str.format('Origin: {}\n'.format(' '.join(origin_sen)))
                        )

                    hypothesis = ' '.join(decoded_words)
                    reference = ' '.join(origin_sen)
                    #there may be several references

                    # BLEU Score Calculation
                    bleu_score_1_ = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights=(1, 0, 0, 0))
                    bleu_score_2_ = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights=(0, 1, 0, 0))
                    bleu_score_3_ = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights=(0, 0, 1, 0))
                    bleu_score_4_ = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights=(0, 0, 0, 1))
                    
                    sentence_bleu_score = [bleu_score_1_, bleu_score_2_, bleu_score_3_, bleu_score_4_]

                    for num, val in enumerate(sentence_bleu_score):
                        generate_text = (
                            generate_text + 
                            str.format('BLEU-{}: {}\n'.format(
                                (num+1), 
                                val
                                )
                            )
                        )    

                    bleu_score_1 += bleu_score_1_
                    bleu_score_2 += bleu_score_2_
                    bleu_score_3 += bleu_score_3_
                    bleu_score_4 += bleu_score_4_

                    if Epoch >3:
                        # ROUGE Score Calculation
                        try:
                            _rouge_score_current = rouge.get_scores(hypothesis, reference)[0]
                            for _rouge_method, _metrics in _rouge_score_current.items():
                                for _key, _val in _metrics.items():
                                    _rouge_score[_rouge_method][_key] += _val                         
                            pass
                        except Exception as msg:
                            print(msg)
                            stop= 1
                            pass
                    
                    
                   

                    # Write down sentences
                    if _write_mode =='generate':
                        if self.test_on_train_data :
                            fpath = (R'{}/GenerateSentences/on_train/'.format(self.save_dir))
                        else:
                            fpath = (R'{}/GenerateSentences/on_test/'.format(self.save_dir))

                        with open(fpath + 'sentences_ep{}.txt'.format(self.training_epoch),'a') as file:
                            file.write(generate_text)  

                        if (write_insert_sql):
                            # Write insert sql
                            sqlpath = (fpath + 'insert.sql')
                            self._write_generate_reviews_into_sqlfile(
                                sqlpath, 
                                self.userObj.index2reviewerID[user_.item()],
                                self.itemObj.index2asin[asin_.item()],
                                ' '.join(decoded_words)
                                )   
                
                # Average bleu score through reviewer
                batch_bleu_score_1 += (bleu_score_1/len(current_reviewerIDs))
                batch_bleu_score_2 += (bleu_score_2/len(current_reviewerIDs))
                batch_bleu_score_3 += (bleu_score_3/len(current_reviewerIDs))
                batch_bleu_score_4 += (bleu_score_4/len(current_reviewerIDs))

                if Epoch >3:
                    # Average rouge score through reviewer
                    for _rouge_method, _metrics in _rouge_score.items():
                        for _key, _val in _metrics.items():
                            average_rouge_score[_rouge_method][_key] += (_val/len(current_reviewerIDs))


        num_of_iter = len(self.testing_batches[0])*len(self.testing_batch_labels)
        
        RMSE = group_loss/num_of_iter
        batch_bleu_score = [
            batch_bleu_score_1/num_of_iter, 
            batch_bleu_score_2/num_of_iter, 
            batch_bleu_score_3/num_of_iter, 
            batch_bleu_score_4/num_of_iter
            ]
        
        _nllloss = decoder_epoch_loss/num_of_iter
        
        if Epoch >3:
            for _rouge_method, _metrics in average_rouge_score.items():
                for _key, _val in _metrics.items():
                    average_rouge_score[_rouge_method][_key] = _val/num_of_iter


        return RMSE, _nllloss, batch_bleu_score, average_rouge_score


    def _train_iteration_nrt(self, NRT, NRTD, NRT_optimizer, NRTD_optimizer,
        training_batches, external_memorys, candidate_items, 
        candidate_users, training_batch_labels, label_sentences, voc=None,
        fix_rating=False
        ):
        """
        Method for baseline model (NRT)
        """        

        # Initialize this epoch loss
        nrt_epoch_loss = 0
        decoder_epoch_loss = 0

        for batch_ctr in tqdm.tqdm(range(len(training_batches[0]))): # amount of batches
            for idx in range(len(training_batch_labels)):
                # Forward pass through nrt
                NRT_optimizer.zero_grad()
                NRTD_optimizer.zero_grad()

                
                target_batch = label_sentences[0][batch_ctr]
                # Ground true sentences & rating
                target_variable, target_len, ratings = target_batch 
                target_variable = target_variable.to(self.device)
                max_target_len = max(target_len)

                current_asins = torch.tensor(candidate_items[idx][batch_ctr]).to(self.device)
                current_reviewerIDs = torch.tensor(candidate_users[idx][batch_ctr]).to(self.device)

                # Pass through NTR
                rating_output, content_output = NRT(current_asins, current_reviewerIDs)                    
                rating_output = rating_output.squeeze(1)

                # Caculate Square Loss 
                current_labels = torch.tensor(training_batch_labels[idx][batch_ctr]).to(self.device)
                rating_output_ = (rating_output*(5-1)+1)
                err = rating_output_ - current_labels

                nrt_loss = torch.mul(err, err)
                nrt_loss = torch.mean(nrt_loss, dim=0)

                # NRT loss of this epoch
                nrt_epoch_loss += nrt_loss                
                rating_output_ = torch.round(rating_output_, out=None).long()


                # Calculate content loss
                _word_term_freq = torch.FloatTensor(self._word_term_freq).to(self.device)
                _content_loss = 0

                for _row in content_output:
                    c_loss = _word_term_freq * torch.log(_row)
                    _row_content_loss = torch.sum(c_loss)
                    _content_loss += _row_content_loss
                    pass

                _content_loss = _content_loss/len(content_output)


                """
                Runing Decoder
                """
                # Create initial decoder input (start with SOS tokens for each sentence)
                decoder_input = torch.LongTensor([[self.SOS_token for _ in range(self.batch_size)]])
                decoder_input = decoder_input.to(self.device)   

                # Set initial decoder hidden state to the inter_hidden's final hidden state
                criterion = nn.NLLLoss()
                decoder_loss = 0
                decoder_hidden = None   # initial

                # Determine if we are using teacher forcing this iteration
                use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

                # Forward batch of sequences through decoder one time step at a time
                if use_teacher_forcing:
                    for t in range(max_target_len):
                        if(t==0):
                            initial = True
                        else:
                            initial = False

                        decoder_output, decoder_hidden = NRTD(
                            decoder_input, 
                            current_asins, 
                            current_reviewerIDs, 
                            rating_output_,
                            decoder_hidden,
                            initial = initial
                        )
                        # Teacher forcing: next input is current target
                        decoder_input = target_variable[t].view(1, -1)  # get the row(word) of sentences

                        # Calculate and accumulate loss
                        nll_loss = criterion(decoder_output, target_variable[t])
                        decoder_loss += nll_loss
                else:
                    for t in range(max_target_len):
                        if(t==0):
                            initial = True
                        else:
                            initial = False

                        decoder_output, decoder_hidden = NRTD(
                            decoder_input, 
                            current_asins, 
                            current_reviewerIDs, 
                            rating_output_,
                            decoder_hidden,
                            initial = initial
                        )
                        # No teacher forcing: next input is decoder's own current output
                        _, topi = decoder_output.topk(1)

                        decoder_input = torch.LongTensor([[topi[i][0] for i in range(self.batch_size)]])
                        decoder_input = decoder_input.to(self.device)

                        # Calculate and accumulate loss
                        nll_loss = criterion(decoder_output, target_variable[t])
                        decoder_loss += nll_loss
                    pass
                
                # loss = nrt_loss + decoder_loss
                loss = nrt_loss + (decoder_loss/max_target_len) + _content_loss
                # Perform backpropatation
                loss.backward()

                # Clip gradients: gradients are modified in place
                _ = nn.utils.clip_grad_norm_(NRTD.parameters(), self.clip)
                _ = nn.utils.clip_grad_norm_(NRT.parameters(), self.clip)
                NRTD_optimizer.step()
                NRT_optimizer.step()

                # decoder loss of this epoch
                decoder_epoch_loss += decoder_loss.item()/float(max_target_len)

        return nrt_epoch_loss, decoder_epoch_loss

    def calculate_word_frequency(self):
        """ 
        Calculate word frequency for coverage mechanism
        """
        _word_term_freq = list()
        _total_count = 0
        
        for _word, _count in self.voc.word2count.items():
            _total_count += _count

        for _word, _count in self.voc.word2count.items():
            _word_term_freq.append(
                float(_count/_total_count)
            )
        
        return _word_term_freq

    def train_nrt(self, select_table, isStoreModel=False, ep_to_store=0, WriteTrainLoss=False, store_every = 2, use_pretrain_item= False, pretrain_wordVec=None , voc=None):
        
        asin, reviewerID = self._get_asin_reviewer()
        # Initialize textual embeddings
        if(pretrain_wordVec != None):
            embedding = pretrain_wordVec
        else:
            embedding = nn.Embedding(self.voc.num_words, self.hidden_size)

        # Calculate word freq
        self._word_term_freq = self.calculate_word_frequency()

        # Initialize asin/reviewer embeddings
        if(use_pretrain_item):
            asin_embedding = torch.load(R'PretrainingEmb/item_embedding_fromGRU.pth')
        else:
            asin_embedding = nn.Embedding(len(asin), self.hidden_size)
        reviewerID_embedding = nn.Embedding(len(reviewerID), self.hidden_size)   

        rating_embedding = nn.Embedding(5 , self.hidden_size)

        NRT = nrt_rating_predictor(self.hidden_size, asin_embedding, reviewerID_embedding, self.voc.num_words)
        NRT = NRT.to(self.device)
        NRT.train()

        NRTD = nrt_decoder(self.hidden_size, self.voc.num_words, asin_embedding, reviewerID_embedding, rating_embedding, embedding)
        NRTD = NRTD.to(self.device)
        NRTD.train()

        # Initialize optimizers    
        NRT_optimizer = optim.AdamW(
            NRT.parameters(), 
            lr=self.learning_rate,
            weight_decay=0.001
            )

        NRTD_optimizer = optim.AdamW(
            NRTD.parameters(), 
            lr=self.learning_rate * self.decoder_learning_ratio, 
            weight_decay=0.001
            )

        print('Models built and ready to go!')        

        for Epoch in range(self.training_epoch):
            
            nrt_loss_average = 99
            fix_rating = True if nrt_loss_average < 1.089 else False
            
            # Run a training iteration with batch
            nrt_group_loss, decoder_group_loss = self._train_iteration_nrt(
                NRT, NRTD, NRT_optimizer, NRTD_optimizer,
                self.training_batches, self.external_memorys, 
                self.candidate_items, self.candidate_users, 
                self.training_batch_labels, self.label_sentences,
                voc = voc, fix_rating=fix_rating
                )

            # num_of_iter = len(self.training_batches[0])*len(self.training_batch_labels)*len(self.training_batches)
            num_of_iter = len(self.training_batches[0])*len(self.training_batch_labels)
        
            nrt_loss_average = nrt_group_loss/num_of_iter
            decoder_loss_average = decoder_group_loss/num_of_iter

            print('Epoch:{}\tNRT(SE):{}\tNNL:{}\t'.format(Epoch, nrt_loss_average, decoder_loss_average))

            if(Epoch % store_every == 0 and isStoreModel and Epoch >= ep_to_store):
                torch.save(NRT, R'{}/Model/NRT_epoch{}'.format(self.save_dir, Epoch))
                torch.save(NRTD, R'{}/Model/NRTD_epoch{}'.format(self.save_dir, Epoch))
    
            if WriteTrainLoss:
                with open(R'{}/Loss/TrainingLoss.txt'.format(self.save_dir),'a') as file:
                    file.write('Epoch:{}\tNRT(SE):{}\tNNL:{}\n'.format(Epoch, nrt_loss_average, decoder_loss_average))
            
            # evaluating
            RMSE, batch_bleu_score, average_rouge_score = self.evaluate_nrt(
                NRT, 
                NRTD, 
                self.label_sentences,
                voc=voc, 
                write_insert_sql=False, 
                candidateObj=None
                )

            # Write Bleu score
            for num, val in enumerate(batch_bleu_score):
                with open('{}/Bleu/blue{}.score.txt'.format(self.save_dir, (num+1)),'a') as file:
                    file.write('BLEU SCORE {}.ep.{}: {}\n'.format((num+1), Epoch, val))
                print('BLEU SCORE {}: {}'.format((num+1), val))

            # Write MSE
            print('Epoch:{}\tMSE:{}\t'.format(Epoch, RMSE))
            with open(R'{}/Loss/TestingLoss.txt'.format(self.save_dir),'a') as file:
                file.write('Epoch:{}\tRMSE:{}\n'.format(Epoch, RMSE))   


            with open('{}/Bleu/rouge.score.txt'.format(self.save_dir), 'a') as file:
                file.write('=============================\nEpoch:{}\n'.format(Epoch))
                for _rouge_method, _metrics in average_rouge_score.items():
                    for _key, _val in _metrics.items():
                        file.write('{}. {}: {}\n'.format(_rouge_method, _key, _val))
                        print('{}. {}: {}'.format(_rouge_method, _key, _val))

        pass

    def evaluate_nrt(self, NRT, NRTD, label_sentences, voc=None, write_insert_sql=False, candidateObj=None):

        # Initialize this epoch loss
        nrt_epoch_loss = 0
        batch_bleu_score_1 = 0
        batch_bleu_score_2 = 0
        batch_bleu_score_3 = 0
        batch_bleu_score_4 = 0
        
        rouge = Rouge()

        average_rouge_score = {
            'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0},
            'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
            'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}
            }              

        for batch_ctr in range(len(self.testing_batches[0])): #how many batches
            for idx in range(len(self.testing_batch_labels)):
                # Forward pass through nrt
                # for reviews_ctr in range(len(self.testing_batches)): # iter. through reviews

                if True:
                    reviews_ctr = len(self.testing_batches)-1

                    # current_batch = self.testing_batches[reviews_ctr][batch_ctr]

                    target_batch = self.testing_label_sentences[0][batch_ctr]
                    # Ground true sentences & rating
                    target_variable, target_len, ratings = target_batch 
                    target_variable = target_variable.to(self.device)
                    max_target_len = max(target_len)
                    target_len = target_len.to(self.device)                  

                    current_asins = torch.tensor(self.testing_asins[idx][batch_ctr]).to(self.device)
                    current_reviewerIDs = torch.tensor(self.testing_reviewerIDs[idx][batch_ctr]).to(self.device)

                    # Pass through NTR
                    rating_output, content_output = NRT(current_asins, current_reviewerIDs)                    
                    rating_output = rating_output.squeeze(1)

                    # Caculate Square Loss 
                    current_labels = torch.tensor(self.testing_batch_labels[idx][batch_ctr]).to(self.device)
                    rating_output_ = (rating_output*(5-1)+1)
                    err = rating_output_ - current_labels

                    nrt_loss = torch.mul(err, err)
                    nrt_loss = torch.mean(nrt_loss, dim=0)
                    nrt_loss = torch.sqrt(nrt_loss)

                    # NRT loss of this epoch
                    nrt_epoch_loss += nrt_loss                
                    rating_output_ = torch.round(rating_output_, out=None).long()

                    """
                    Runing Decoder
                    """
                    # Create initial decoder input (start with SOS tokens for each sentence)
                    decoder_input = torch.LongTensor([[self.SOS_token for _ in range(self.batch_size)]])
                    decoder_input = decoder_input.to(self.device)   

                    decoder_hidden = None   # initial decoder hidden

                    # Initialize tensors to append decoded words to
                    all_tokens = torch.zeros([0], device=self.device, dtype=torch.long)
                    all_scores = torch.zeros([0], device=self.device)  

                    for t in range(max_target_len):
                        if(t==0):
                            initial = True
                        else:
                            initial = False

                        decoder_output, decoder_hidden = NRTD(
                            decoder_input, 
                            current_asins, 
                            current_reviewerIDs, 
                            rating_output_,
                            decoder_hidden,
                            initial = initial
                        )
                        # No teacher forcing: next input is decoder's own current output
                        decoder_scores_, topi = decoder_output.topk(1)
                        decoder_input = torch.LongTensor([[topi[i][0] for i in range(self.batch_size)]])
                        decoder_input = decoder_input.to(self.device)

                        ds, di = torch.max(decoder_output, dim=1)

                        # Record token and score
                        all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
                        all_scores = torch.cat((all_scores, torch.t(decoder_scores_)), dim=0) 


                    """
                    Decode user review from search result.
                    """
                    bleu_score_1 = 0
                    bleu_score_2 = 0
                    bleu_score_3 = 0
                    bleu_score_4 = 0

                    _rouge_score = {
                        'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                        'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                        'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}
                        }                    

                    for index_ , user_ in enumerate(current_reviewerIDs):
                        
                        asin_ = current_asins[index_]

                        current_user_tokens = all_tokens[:,index_].tolist()
                        decoded_words = [voc.index2word[token] for token in current_user_tokens if token != 0]

                        
                        current_user_sen = target_variable[:,index_].tolist()
                        origin_sen = [voc.index2word[token] for token in current_user_sen if token != 0]                        

                        hypothesis = ' '.join(decoded_words)
                        reference = ' '.join(origin_sen)
                        
                        # BLEU Score Calculation
                        bleu_score_1_ = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights=(1, 0, 0, 0))
                        bleu_score_2_ = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights=(0, 1, 0, 0))
                        bleu_score_3_ = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights=(0, 0, 1, 0))
                        bleu_score_4_ = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights=(0, 0, 0, 1))
                        sentence_bleu_score = [bleu_score_1_, bleu_score_2_, bleu_score_3_, bleu_score_4_]

                        bleu_score_1 += bleu_score_1_
                        bleu_score_2 += bleu_score_2_
                        bleu_score_3 += bleu_score_3_
                        bleu_score_4 += bleu_score_4_

                        # ROUGE Score Calculation
                        try:
                            _rouge_score_current = rouge.get_scores(hypothesis, reference)[0]
                            for _rouge_method, _metrics in _rouge_score_current.items():
                                for _key, _val in _metrics.items():
                                    _rouge_score[_rouge_method][_key] += _val      
                            pass                           
                        except ValueError as msg:

                            pass
                                     
                    # Average bleu score through reviewer
                    batch_bleu_score_1 += (bleu_score_1/len(current_reviewerIDs))
                    batch_bleu_score_2 += (bleu_score_2/len(current_reviewerIDs))
                    batch_bleu_score_3 += (bleu_score_3/len(current_reviewerIDs))
                    batch_bleu_score_4 += (bleu_score_4/len(current_reviewerIDs))       

                    # Average rouge score through reviewer
                    for _rouge_method, _metrics in _rouge_score.items():
                        for _key, _val in _metrics.items():
                            average_rouge_score[_rouge_method][_key] += (_val/len(current_reviewerIDs))                                 
                    
        num_of_iter = len(self.testing_batches[0])*len(self.testing_batch_labels)
        
        RMSE = nrt_epoch_loss/num_of_iter
        batch_bleu_score = [
            batch_bleu_score_1/num_of_iter, 
            batch_bleu_score_2/num_of_iter, 
            batch_bleu_score_3/num_of_iter, 
            batch_bleu_score_4/num_of_iter
            ]

        for _rouge_method, _metrics in average_rouge_score.items():
            for _key, _val in _metrics.items():
                average_rouge_score[_rouge_method][_key] = _val/num_of_iter            

        return RMSE, batch_bleu_score, average_rouge_score
    
    def evaluate_nrt_generation(self, NRT, NRTD):

        for batch_ctr in range(len(self.testing_batches[0])): #how many batches
            for idx in range(len(self.testing_batch_labels)):
                # Forward pass through nrt

                target_batch = self.testing_label_sentences[0][batch_ctr]
                # Ground true sentences & rating
                target_variable, target_len, ratings = target_batch 
                target_variable = target_variable.to(self.device)
                max_target_len = max(target_len)
                target_len = target_len.to(self.device)                  

                current_asins = torch.tensor(self.testing_asins[idx][batch_ctr]).to(self.device)
                current_reviewerIDs = torch.tensor(self.testing_reviewerIDs[idx][batch_ctr]).to(self.device)

                # Pass through NTR
                rating_output, content_output = NRT(current_asins, current_reviewerIDs)                    
                rating_output = rating_output.squeeze(1)

                # Caculate Square Loss 
                current_labels = torch.tensor(self.testing_batch_labels[idx][batch_ctr]).to(self.device)
                rating_output_ = (rating_output*(5-1)+1)             
                rating_output_ = torch.round(rating_output_, out=None).long()

                """
                Runing Decoder
                """
                # Create initial decoder input (start with SOS tokens for each sentence)
                decoder_input = torch.LongTensor([[self.SOS_token for _ in range(self.batch_size)]])
                decoder_input = decoder_input.to(self.device)   

                decoder_hidden = None   # initial decoder hidden

                # Initialize tensors to append decoded words to
                all_tokens = torch.zeros([0], device=self.device, dtype=torch.long)
                all_scores = torch.zeros([0], device=self.device)  

                for t in range(max_target_len):
                    if(t==0):
                        initial = True
                    else:
                        initial = False

                    decoder_output, decoder_hidden = NRTD(
                        decoder_input, 
                        current_asins, 
                        current_reviewerIDs, 
                        rating_output_,
                        decoder_hidden,
                        initial = initial
                    )
                    # No teacher forcing: next input is decoder's own current output
                    decoder_scores_, topi = decoder_output.topk(1)
                    decoder_input = torch.LongTensor([[topi[i][0] for i in range(self.batch_size)]])
                    decoder_input = decoder_input.to(self.device)

                    ds, di = torch.max(decoder_output, dim=1)

                    # Record token and score
                    all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
                    all_scores = torch.cat((all_scores, torch.t(decoder_scores_)), dim=0) 


                """
                Decode user review from search result.
                """              
                for index_ , user_ in enumerate(current_reviewerIDs):
                    
                    asin_ = current_asins[index_]

                    current_user_tokens = all_tokens[:,index_].tolist()
                    decoded_words = [self.voc.index2word[token] for token in current_user_tokens if token != 0]

                    
                    current_user_sen = target_variable[:,index_].tolist()
                    origin_sen = [self.voc.index2word[token] for token in current_user_sen if token != 0]

                    try:
                        product_title = self.asin2title[
                            self.itemObj.index2asin[asin_.item()]
                        ]
                    except Exception as ex:
                        product_title = 'None'
                        pass


                    hypothesis = ' '.join(decoded_words)
                    reference = ' '.join(origin_sen)

                    generate_text = str.format(
f"""
=========================
Userid & asin:{self.userObj.index2reviewerID[user_.item()]},{self.itemObj.index2asin[asin_.item()]}
title:{product_title}
Predict:{(rating_output*(5-1)+1)[index_].item()}
Rating:{current_labels[index_].item()}
Generate: {hypothesis}
Origin: {reference}
"""
                    )


                    # BLEU Score Calculation
                    bleu_score_1_ = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights=(1, 0, 0, 0))
                    bleu_score_2_ = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights=(0, 1, 0, 0))
                    bleu_score_3_ = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights=(0, 0, 1, 0))
                    bleu_score_4_ = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights=(0, 0, 0, 1))
                    
                    sentence_bleu_score = [bleu_score_1_, bleu_score_2_, bleu_score_3_, bleu_score_4_]

                    for num, val in enumerate(sentence_bleu_score):
                        generate_text = (
                            generate_text + 
                            str.format('BLEU-{}: {}\n'.format(
                                (num+1), 
                                val
                                )
                            )
                        )    



                    fpath = (R'{}/GenerateSentences/on_test/'.format(self.save_dir))
                    with open(fpath + 'sentences_ep{}.txt'.format(self.training_epoch),'a') as file:
                        file.write(generate_text)              

        return 0

    def set_testing_set(self, test_on_train_data='Y'):
        if test_on_train_data == 'Y':
            self.test_on_train_data = True
        elif test_on_train_data == 'N':
            self.test_on_train_data = False

    def evaluate_mse(self, IntraGRU, InterGRU, isCatItemVec=False, concat_rating=False, isWriteAttn=False):    
        
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
            
                    # Concat. asin feature
                    this_asins = self.testing_external_memorys[reviews_ctr][batch_ctr]
                    this_asins = torch.tensor([val for val in this_asins]).to(self.device)
                    this_asins = this_asins.unsqueeze(0)

                    if(concat_rating):
                        this_rating = self.testing_review_rating[reviews_ctr][batch_ctr]
                        
                        _encode_rating = self._rating_to_onehot(this_rating)
                        _encode_rating = torch.tensor(_encode_rating).to(self.device)
                        _encode_rating = _encode_rating.unsqueeze(0)
                        pass
                    else:
                        inter_intput_rating =None

                    with torch.no_grad():
                        outputs, intra_hidden, intra_attn_score = IntraGRU[reviews_ctr](input_variable, lengths, 
                            current_asins, current_reviewerIDs)
                        outputs = outputs.unsqueeze(0)

                        if(reviews_ctr == 0):
                            interInput = outputs
                            interInput_asin = this_asins
                            if(concat_rating):
                                inter_intput_rating = _encode_rating                                
                        else:
                            interInput = torch.cat((interInput, outputs) , 0) 
                            interInput_asin = torch.cat((interInput_asin, this_asins) , 0) 
                            if(concat_rating):
                                inter_intput_rating = torch.cat(
                                    (inter_intput_rating, _encode_rating) , 0
                                    )                             

                                    
                    with torch.no_grad():
                        outputs, intra_hidden, inter_attn_score, context_vector  = InterGRU(
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

    def _write_generate_reviews_into_sqlfile(self, fpath, reviewerID, asin, generative_review, table='#table'):
        """
        Store the generative result into sql format file.
        """
        sql = (
            """
            INSERT INTO {} 
            (`reviewerID`, `asin`, `generative_review`) VALUES 
            ('{}', '{}', '{}');
            """.format(table, reviewerID, asin, generative_review)
        )

        with open(fpath,'a') as file:
            file.write(sql) 
        pass


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