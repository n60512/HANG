import torch
import torch.nn as nn
import torch.nn.functional as F

class IntraReviewGRU(nn.Module):
    def __init__(self, hidden_size, embedding, itemEmbedding, userEmbedding, n_layers=1, dropout=0, latentK = 64, method = 'dualFC'):
        super(IntraReviewGRU, self).__init__()
        
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = embedding
        self.itemEmbedding = itemEmbedding
        self.userEmbedding = userEmbedding
        self.method = method
    

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
            self.linear_alpha = torch.nn.Linear(hidden_size, 1) 

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

        elif self.method == 'dualFC':
            self.linear1 = torch.nn.Linear(hidden_size, hidden_size)
            self.linear2 = torch.nn.Linear(hidden_size, hidden_size)
            self.linear_alpha = torch.nn.Linear(hidden_size, 1)       

        self.intra_review = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), 
                          bidirectional=True)
                         
    def CalculateAttn(self, key_vector, query_vector):
        
        # Calculate weighting score
        if(self.method == 'dualFC'):
            x = F.relu(self.linear1(key_vector) +
                    self.linear2(query_vector) 
                )
            weighting_score = self.linear_alpha(x)
            # Calculate attention score
            intra_attn_score = torch.softmax(weighting_score, dim = 0)

        elif (self.method=='dot'):
            intra_attn_score = key_vector * query_vector
            
        elif (self.method=='general'):
            energy = self.attn(query_vector)
            x = F.relu(key_vector * energy)
            weighting_score = self.linear_alpha(x)
            # Calculate attention score            
            intra_attn_score = torch.softmax(weighting_score, dim = 0)

        return intra_attn_score
        
    def forward(self, input_seq, input_lengths, item_index, user_index, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)           
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, enforce_sorted=False)
        # Forward pass through GRU
        outputs, hidden = self.intra_review(packed, hidden)
 
        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)

        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]

        # Calculate element-wise product
        elm_w_product = self.itemEmbedding(item_index) * self.userEmbedding(user_index)

        # Calculate attention score

        if self.method != 'without':
            intra_attn_score = self.CalculateAttn(outputs, elm_w_product)
            new_outputs = intra_attn_score * outputs
            intra_outputs = torch.sum(new_outputs , dim = 0)    # output sum
        else:
            intra_outputs = torch.sum(outputs , dim = 0)    # output sum
            intra_attn_score = None

        # Return output and final hidden state
        return intra_outputs, hidden, intra_attn_score

class HANN(nn.Module):
    def __init__(self, hidden_size, embedding, itemEmbedding, userEmbedding, n_layers=1, dropout=0, latentK = 64, 
        isCatItemVec=False, netType='item_base', method='dualFC'):
        super(HANN, self).__init__()
        
        self.method = method
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.latentK = latentK

        self.embedding = embedding
        self.itemEmbedding = itemEmbedding
        self.userEmbedding = userEmbedding

        self.isCatItemVec = isCatItemVec
        self.netType = netType

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
            self.linear_beta = torch.nn.Linear(hidden_size, 1)   
            
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

        elif self.method == 'dualFC':            
            self.linear3 = torch.nn.Linear(hidden_size, hidden_size)
            self.linear4 = torch.nn.Linear(hidden_size, hidden_size)
            self.linear_beta = torch.nn.Linear(hidden_size, 1)      

        if(self.isCatItemVec):
            GRU_InputSize = hidden_size*2
            # GRU_InputSize = hidden_size + latentK   # word dim. + item dim.
        else:
            GRU_InputSize = hidden_size

        self.inter_review = nn.GRU(GRU_InputSize, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout))
                          
        self.dropout = nn.Dropout(dropout)
        
        self.fc_doubleK = nn.Linear(hidden_size*2 , self.latentK*2)
        # self.fc_doubleK = nn.Linear(hidden_size + latentK , self.latentK*2)
        self.fc_singleK = nn.Linear(self.latentK*2, self.latentK)
        self.fc_out = nn.Linear(self.latentK, 1)
    
    def CalculateAttn(self, key_vector, query_vector):
        
        # Calculate weighting score
        if(self.method == 'dualFC'):
            x = F.relu(self.linear3(key_vector) +
                    self.linear4(query_vector) 
                )
            weighting_score = self.linear_beta(x)
            # Calculate attention score
            inter_attn_score = torch.softmax(weighting_score, dim = 0)

        elif (self.method=='dot'):
            inter_attn_score = key_vector * query_vector
            
        elif (self.method=='general'):
            energy = self.attn(query_vector)
            x = F.relu(key_vector * energy)
            weighting_score = self.linear_beta(x)
            # Calculate attention score            
            inter_attn_score = torch.softmax(weighting_score, dim = 0)

        return inter_attn_score

    def forward(self, intra_outputs, this_candidate_index, item_index, user_index, hidden=None):
        
        if(self.isCatItemVec):
            # Concat. intra output && candidate feature
            if(self.netType=='user_base'):
                item_feature = self.itemEmbedding(this_candidate_index)
                inter_input = torch.cat((intra_outputs, item_feature), 2)
            elif(self.netType=='item_base'):
                user_feature = self.userEmbedding(this_candidate_index)
                inter_input = torch.cat((intra_outputs, user_feature), 2)            
        else:
            inter_input = intra_outputs

        # Forward pass through GRU
        outputs, hidden = self.inter_review(inter_input, hidden)

        # Calculate element-wise product
        elm_w_product_inter = self.itemEmbedding(item_index) * self.userEmbedding(user_index)

        # a = outputs * elm_w_product_inter
        # Calculate attention score
        inter_attn_score = self.CalculateAttn(outputs, elm_w_product_inter)

        # Consider attention score
        weighting_outputs = inter_attn_score * outputs
        context_vector = weighting_outputs

        outputs_sum = torch.sum(weighting_outputs , dim = 0)  

        # Concat. interaction vector & GRU output
        outputs_cat = torch.cat((outputs_sum, elm_w_product_inter), dim=1)
        
        # dropout
        outputs_cat = self.dropout(outputs_cat)

        # hidden_size to 2*K dimension
        outputs_ = self.fc_doubleK(outputs_cat) 
        # 2*K to K dimension
        outputs_ = self.fc_singleK(outputs_)  
        # K to 1 dimension
        outputs_ = self.fc_out(outputs_)

        sigmoid_outputs = torch.sigmoid(outputs_)
        # sigmoid_outputs = torch.tanh(outputs_)
        sigmoid_outputs = sigmoid_outputs.squeeze(0)

        # Return output and final hidden state
        return sigmoid_outputs, hidden, inter_attn_score, context_vector

class HANN_new(nn.Module):
    def __init__(self, hidden_size, embedding, itemEmbedding, userEmbedding, n_layers=1, dropout=0, latentK = 64, 
        isCatItemVec=False, concat_rating=False, netType='item_base', method='dualFC'):
        super(HANN_new, self).__init__()
        
        self.method = method
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.latentK = latentK

        self.embedding = embedding
        self.itemEmbedding = itemEmbedding
        self.userEmbedding = userEmbedding

        self.isCatItemVec = isCatItemVec
        self.concat_rating = concat_rating
        self.netType = netType

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
            self.linear_beta = torch.nn.Linear(hidden_size, 1)   
            
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

        elif self.method == 'dualFC':            
            self.linear3 = torch.nn.Linear(hidden_size, hidden_size)
            self.linear4 = torch.nn.Linear(hidden_size, hidden_size)
            self.linear_beta = torch.nn.Linear(hidden_size, 1)      

        if(self.isCatItemVec):
            GRU_InputSize = hidden_size*2
            # GRU_InputSize = hidden_size + latentK   # word dim. + item dim.
        else:
            GRU_InputSize = hidden_size

        if(self.concat_rating):
            GRU_InputSize = GRU_InputSize+5                

        self.inter_review = nn.GRU(GRU_InputSize, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout))
                          
        self.dropout = nn.Dropout(dropout)
        
        self.fc_doubleK = nn.Linear(hidden_size*2 , self.latentK*2)
        # self.fc_doubleK = nn.Linear(hidden_size + latentK , self.latentK*2)
        self.fc_singleK = nn.Linear(self.latentK*2, self.latentK)
        self.fc_out = nn.Linear(self.latentK, 1)
    
    def CalculateAttn(self, key_vector, query_vector):
        
        # Calculate weighting score
        if(self.method == 'dualFC'):
            x = F.relu(self.linear3(key_vector) +
                    self.linear4(query_vector) 
                )
            weighting_score = self.linear_beta(x)
            # Calculate attention score
            inter_attn_score = torch.softmax(weighting_score, dim = 0)

        elif (self.method=='dot'):
            inter_attn_score = key_vector * query_vector
            
        elif (self.method=='general'):
            energy = self.attn(query_vector)
            x = F.relu(key_vector * energy)
            weighting_score = self.linear_beta(x)
            # Calculate attention score            
            inter_attn_score = torch.softmax(weighting_score, dim = 0)

        return inter_attn_score

    def forward(self, intra_outputs, this_candidate_index, item_index, user_index, hidden=None, review_rating=None):
        
        if(self.isCatItemVec):
            # Concat. intra output && candidate feature
            if(self.netType=='user_base'):
                item_feature = self.itemEmbedding(this_candidate_index)
                inter_input = torch.cat((intra_outputs, item_feature), 2)
            elif(self.netType=='item_base'):
                user_feature = self.userEmbedding(this_candidate_index)
                inter_input = torch.cat((intra_outputs, user_feature), 2)            
        else:
            inter_input = intra_outputs

        if(self.concat_rating):
            inter_input = torch.cat((inter_input, review_rating), 2)

        # Forward pass through GRU
        outputs, hidden = self.inter_review(inter_input, hidden)

        # Calculate element-wise product
        elm_w_product_inter = self.itemEmbedding(item_index) * self.userEmbedding(user_index)

        # Calculate attention score
        inter_attn_score = self.CalculateAttn(outputs, elm_w_product_inter)

        # Consider attention score
        weighting_outputs = inter_attn_score * outputs
        context_vector = weighting_outputs

        outputs_sum = torch.sum(weighting_outputs , dim = 0)  

        # Concat. interaction vector & GRU output
        outputs_cat = torch.cat((outputs_sum, elm_w_product_inter), dim=1)
        
        # dropout
        outputs_cat = self.dropout(outputs_cat)

        # hidden_size to 2*K dimension
        outputs_ = self.fc_doubleK(outputs_cat) 
        # 2*K to K dimension
        outputs_ = self.fc_singleK(outputs_)  
        # K to 1 dimension
        outputs_ = self.fc_out(outputs_)

        sigmoid_outputs = torch.sigmoid(outputs_)
        # sigmoid_outputs = torch.tanh(outputs_)
        sigmoid_outputs = sigmoid_outputs.squeeze(0)

        # Return output and final hidden state
        return sigmoid_outputs, hidden, inter_attn_score, context_vector

class DecoderGRU(nn.Module):
    def __init__(self, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(DecoderGRU, self).__init__()

        # self.dec_merge_rating = dec_merge_rating
        self._use_coverage = False

        # Keep for reference
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)

        # Drop out
        self._dropout = nn.Dropout(dropout)

        # merge rating 
        self.weight_rating = nn.Linear(5, hidden_size)
        self.weight_enc_output = nn.Linear(hidden_size, hidden_size)
        self.weight_user = nn.Linear(hidden_size, hidden_size)
        self.weight_item = nn.Linear(hidden_size, hidden_size)

        self.weight_coverage = nn.Linear(output_size, hidden_size)
        self._fc_300 = nn.Linear(905, hidden_size)

        # GRU model
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.out = nn.Linear(hidden_size, output_size)

        self.logsoftmax = nn.LogSoftmax(dim=1)

        self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)

        self.concat_coverage = nn.Linear(hidden_size * 3, hidden_size)
    
    def set_user_embedding(self, _user_embedding):
        self._user_embedding = _user_embedding

    def set_item_embedding(self, _item_embedding):
        self._item_embedding = _item_embedding

    def CalculateAttn(self, hidden, encoder_output):
        # Linear layer to calculate weighting score
        energy = self.attn(encoder_output)
        weighting_score = torch.sum(hidden * energy, dim=2)
        weighting_score = weighting_score.t()
        
        # Activation function
        attn_weights = torch.softmax(weighting_score, dim=1).unsqueeze(1)

        return attn_weights
    
    def get_softmax_output(self):
        return torch.softmax(self.output, dim=1)
    
    def set_coverage_prob(self, previous_prob, _use_coverage):
        self._use_coverage = _use_coverage
        self.coverage_prob = previous_prob
        pass

    def forward(self, input_step, last_hidden, context_vector, _encode_rating=None , _user_emb=None, _item_emb=None, _enable_attention=True):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        
        # Get user embedding of current input user
        _user_emb = self._user_embedding(_user_emb)
        _user_emb = _user_emb.unsqueeze(0)

        _item_emb = self._item_embedding(_item_emb)
        _item_emb = _item_emb.unsqueeze(0)


        """
        Merge user embedding & rating
        """        
        _MODE = 1   # 0: project to 300d , 1: concat.
        if (_MODE == 0):
            # _MODE : Dense
            initial_hidden = torch.tanh(
                self.weight_rating(_encode_rating) +
                self.weight_enc_output(last_hidden) +
                self.weight_user(_user_emb) +
                self.weight_item(_item_emb) 
            )
        elif (_MODE == 1):
            # _MODE : Concat
            initial_hidden = torch.tanh(
                self._fc_300(torch.cat((last_hidden , _user_emb, _item_emb, _encode_rating), dim=2))
            )

        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, initial_hidden)
        rnn_output = rnn_output.squeeze(0)

        if(_enable_attention):
            attn_weights = self.CalculateAttn(rnn_output, context_vector)
            
            context = attn_weights.bmm(context_vector.transpose(0, 1))
            context = context.squeeze(1)

            # Concat. rnn output & context inf.
            concat_input = torch.cat((rnn_output, context), 1)   
            # Drop out layer
            concat_input = self._dropout(concat_input) 

            if (self._use_coverage):
                # Considerate coverage mechanism
                _coverage_rep = self.weight_coverage(self.coverage_prob)    # previous word probability sum.
                # Concat. attention output & coverage representation
                concat_input = torch.cat((concat_input, _coverage_rep.squeeze_(0)), 1 )
                # FC & Activation
                concat_output = torch.tanh(self.concat_coverage(concat_input))          
                output = self.out(concat_output)
                pass
            else:
                # Origin : without coverage
                concat_output = torch.tanh(self.concat(concat_input))
                output = self.out(concat_output)
                pass
        else:
            # Without attention from Encoder
            rnn_output = torch.tanh(rnn_output)
            output = self.out(rnn_output)
        
        self.output = output

        # log softmax
        output = self.logsoftmax(output)

        # Return output and final hidden state
        return output, hidden

class HANN_i(nn.Module):
    def __init__(self, hidden_size, embedding, itemEmbedding, userEmbedding, n_layers=1, dropout=0, latentK = 64, 
        isCatItemVec=False, concat_rating = False, netType='item_base', method='dualFC'):
        super(HANN_i, self).__init__()
        
        self.method = method
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.latentK = latentK

        self.embedding = embedding
        self.itemEmbedding = itemEmbedding
        self.userEmbedding = userEmbedding

        self.isCatItemVec = isCatItemVec
        self.concat_rating = concat_rating
        self.netType = netType

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
            self.linear_beta = torch.nn.Linear(hidden_size, 1)   
            
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

        elif self.method == 'dualFC':            
            self.linear3 = torch.nn.Linear(hidden_size, hidden_size)
            self.linear4 = torch.nn.Linear(hidden_size, hidden_size)
            self.linear_beta = torch.nn.Linear(hidden_size, 1)      

        if(self.isCatItemVec):
            GRU_InputSize = hidden_size*2
            # GRU_InputSize = hidden_size + latentK   # word dim. + item dim.
        else:
            GRU_InputSize = hidden_size

        if(self.concat_rating):
            GRU_InputSize = GRU_InputSize+5

        self.inter_review = nn.GRU(GRU_InputSize, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout))
                          
        self.dropout = nn.Dropout(dropout)
        
        self.fc_doubleK = nn.Linear(hidden_size*2 , self.latentK*2)
        # self.fc_doubleK = nn.Linear(hidden_size + latentK , self.latentK*2)
        self.fc_singleK = nn.Linear(self.latentK*2, self.latentK)
        self.fc_out = nn.Linear(self.latentK, 1)
    
    def CalculateAttn(self, key_vector, query_vector):
        
        # Calculate weighting score
        if(self.method == 'dualFC'):
            x = F.relu(self.linear3(key_vector) +
                    self.linear4(query_vector) 
                )
            weighting_score = self.linear_beta(x)
            # Calculate attention score
            inter_attn_score = torch.softmax(weighting_score, dim = 0)

        elif (self.method=='dot'):
            inter_attn_score = key_vector * query_vector
            
        elif (self.method=='general'):
            energy = self.attn(query_vector)
            x = F.relu(key_vector * energy)
            weighting_score = self.linear_beta(x)
            # Calculate attention score            
            inter_attn_score = torch.softmax(weighting_score, dim = 0)

        return inter_attn_score

    def forward(self, intra_outputs, this_candidate_index, item_index, user_index, hidden=None, review_rating=None):
        
        if(self.isCatItemVec):
            # Concat. intra output && candidate feature
            if(self.netType=='user_base'):
                item_feature = self.itemEmbedding(this_candidate_index)
                inter_input = torch.cat((intra_outputs, item_feature), 2)
            elif(self.netType=='item_base'):
                user_feature = self.userEmbedding(this_candidate_index)
                inter_input = torch.cat((intra_outputs, user_feature), 2)            
        else:
            inter_input = intra_outputs

        if(self.concat_rating):
            inter_input = torch.cat((inter_input, review_rating), 2)

        # Forward pass through GRU
        outputs, hidden = self.inter_review(inter_input, hidden)

        # Calculate element-wise product
        elm_w_product_inter = self.itemEmbedding(item_index) * self.userEmbedding(user_index)

        # a = outputs * elm_w_product_inter
        # Calculate attention score
        inter_attn_score = self.CalculateAttn(outputs, elm_w_product_inter)

        # Consider attention score
        weighting_outputs = inter_attn_score * outputs
        outputs_sum = torch.sum(weighting_outputs , dim = 0)  

        return outputs_sum, hidden, inter_attn_score

class HANN_u(nn.Module):
    def __init__(self, hidden_size, embedding, itemEmbedding, userEmbedding, n_layers=1, dropout=0, latentK = 64, 
        isCatItemVec=False, concat_rating = False, netType='item_base', method='dualFC'):
        super(HANN_u, self).__init__()
        
        self.method = method
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.latentK = latentK

        self.embedding = embedding
        self.itemEmbedding = itemEmbedding
        self.userEmbedding = userEmbedding

        self.isCatItemVec = isCatItemVec
        self.concat_rating = concat_rating
        self.netType = netType

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
            self.linear_beta = torch.nn.Linear(hidden_size, 1)   
            
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

        elif self.method == 'dualFC':            
            self.linear3 = torch.nn.Linear(hidden_size, hidden_size)
            self.linear4 = torch.nn.Linear(hidden_size, hidden_size)
            self.linear_beta = torch.nn.Linear(hidden_size, 1)      

        if(self.isCatItemVec):
            GRU_InputSize = hidden_size*2
            # GRU_InputSize = hidden_size + latentK   # word dim. + item dim.
        else:
            GRU_InputSize = hidden_size

        if(self.concat_rating):
            GRU_InputSize = GRU_InputSize+5            

        self.inter_review = nn.GRU(GRU_InputSize, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout))
                          
        self.dropout = nn.Dropout(dropout)
        
        self.fc_doubleK = nn.Linear(hidden_size*2 , self.latentK*2)
        # self.fc_doubleK = nn.Linear(hidden_size + latentK , self.latentK*2)
        self.fc_singleK = nn.Linear(self.latentK*2, self.latentK)
        self.fc_out = nn.Linear(self.latentK, 1)
    
    def CalculateAttn(self, key_vector, query_vector):
        
        # Calculate weighting score
        if(self.method == 'dualFC'):
            x = F.relu(self.linear3(key_vector) +
                    self.linear4(query_vector) 
                )
            weighting_score = self.linear_beta(x)
            # Calculate attention score
            inter_attn_score = torch.softmax(weighting_score, dim = 0)

        elif (self.method=='dot'):
            inter_attn_score = key_vector * query_vector
            
        elif (self.method=='general'):
            energy = self.attn(query_vector)
            x = F.relu(key_vector * energy)
            weighting_score = self.linear_beta(x)
            # Calculate attention score            
            inter_attn_score = torch.softmax(weighting_score, dim = 0)

        return inter_attn_score

    def forward(self, intra_outputs, this_candidate_index, item_index, user_index, hidden=None, review_rating=None):
        
        if(self.isCatItemVec):
            # Concat. intra output && candidate feature
            if(self.netType=='user_base'):
                item_feature = self.itemEmbedding(this_candidate_index)
                inter_input = torch.cat((intra_outputs, item_feature), 2)
            elif(self.netType=='item_base'):
                user_feature = self.userEmbedding(this_candidate_index)
                inter_input = torch.cat((intra_outputs, user_feature), 2)            
        else:
            inter_input = intra_outputs

        if(self.concat_rating):
            inter_input = torch.cat((inter_input, review_rating), 2)            

        # Forward pass through GRU
        outputs, hidden = self.inter_review(inter_input, hidden)

        # Calculate element-wise product
        elm_w_product_inter = self.itemEmbedding(item_index) * self.userEmbedding(user_index)

        # a = outputs * elm_w_product_inter
        # Calculate attention score
        inter_attn_score = self.CalculateAttn(outputs, elm_w_product_inter)

        # Consider attention score
        weighting_outputs = inter_attn_score * outputs
        outputs_sum = torch.sum(weighting_outputs , dim = 0)  

        return outputs_sum, hidden, inter_attn_score

class MultiFC(nn.Module):
    def __init__(self, hidden_size, itemEmbedding, userEmbedding, dropout=0, latentK = 64):
        super(MultiFC, self).__init__()
        

        self.hidden_size = hidden_size
        self.latentK = latentK

        self.itemEmbedding = itemEmbedding
        self.userEmbedding = userEmbedding

        self.dropout = nn.Dropout(dropout)
        
        self.fc_doubleK = nn.Linear(hidden_size*3 , self.latentK*2)
        self.fc_singleK = nn.Linear(self.latentK*2, self.latentK)
        self.fc_out = nn.Linear(self.latentK, 1)
    
    def forward(self, item_rep, user_rep, item_index, user_index):
        
        # Calculate element-wise product
        elm_w_product_inter = self.itemEmbedding(item_index) * self.userEmbedding(user_index)

        # Concat. interaction vector & GRU output
        rep_cat = torch.cat((item_rep, user_rep), dim=1)
        rep_cat = torch.cat((rep_cat, elm_w_product_inter), dim=1)
        
        # dropout
        rep_cat = self.dropout(rep_cat)

        # hidden_size to 2*K dimension
        outputs_ = self.fc_doubleK(rep_cat) 
        # 2*K to K dimension
        outputs_ = self.fc_singleK(outputs_)  
        # K to 1 dimension
        outputs_ = self.fc_out(outputs_)

        sigmoid_outputs = torch.sigmoid(outputs_)
        # sigmoid_outputs = torch.tanh(outputs_)
        sigmoid_outputs = sigmoid_outputs.squeeze(0)

        # Return output and final hidden state
        return sigmoid_outputs


class nrt_rating_predictor(nn.Module):
    def __init__(self, hidden_size, itemEmbedding, userEmbedding, voc_size):
        super(nrt_rating_predictor, self).__init__()
        self.itemEmbedding = itemEmbedding
        self.userEmbedding = userEmbedding

        self.weight_user = nn.Linear(hidden_size , hidden_size)
        self.weight_item = nn.Linear(hidden_size , hidden_size)

        self.share_hidden = nn.Linear(hidden_size , hidden_size)
        self.dense = nn.Linear(hidden_size , hidden_size)
        self.out = nn.Linear(hidden_size, 1)

        # the content of review
        self.content_out = nn.Linear(hidden_size, voc_size)

        pass

    def forward(self, item_index, user_index):

        output = self.share_hidden(
            self.weight_user(
                self.userEmbedding(user_index)
            ) + 
            self.weight_item(
                self.itemEmbedding(item_index)
            )
        )

        output = torch.sigmoid(output)
        
        # Adding more layer
        output = torch.sigmoid(
            self.dense(output)
        )
        
        # predict rating
        rating_output = self.out(output)

        # genertate the content of reviews
        content_output = self.content_out(output)
        content_output = torch.softmax(self.content_out(output), dim = 1)
        stop = 1

        return rating_output, content_output

class nrt_decoder(nn.Module):
    def __init__(self, hidden_size, output_size, itemEmbedding, userEmbedding, rating_embedding, embedding, n_layers=1, dropout=0):
        super(nrt_decoder, self).__init__()

        self.itemEmbedding = itemEmbedding
        self.userEmbedding = userEmbedding
        self.rating_embedding = rating_embedding
        self.embedding = embedding
        
        self.weight_user = nn.Linear(hidden_size, hidden_size)
        self.weight_item = nn.Linear(hidden_size, hidden_size)
        self.weight_rate = nn.Linear(hidden_size, hidden_size)

        self.gru = nn.GRU(
            hidden_size, 
            hidden_size, 
            n_layers, 
            dropout=(0 if n_layers == 1 else dropout)
            )

        self.embedding_dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size , hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        
        self.logsoftmax = nn.LogSoftmax(dim=1)

        pass

    def forward(self, input_step, item_index, user_index, rating, last_hidden , initial=False):

        if initial == None:  
            
            # first state (h0)
            last_hidden = torch.tanh(
                self.weight_user(self.userEmbedding(user_index)) + 
                self.weight_item(self.itemEmbedding(item_index)) +
                self.weight_rate(self.rating_embedding(rating))
            )
            last_hidden = last_hidden.unsqueeze(0)

        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)

        rnn_output, hidden = self.gru(embedded, last_hidden)
        rnn_output = rnn_output.squeeze(0)
        rnn_output = self.out(rnn_output)
        
        # log softmax
        output = self.logsoftmax(rnn_output)

        # Return output and final hidden state
        return output, hidden    