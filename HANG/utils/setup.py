class train_test_setup():
    def __init__(self, device, net_type, save_dir, voc, prerocess, 
        training_epoch=100, latent_k=32, batch_size=40, hidden_size=300, clip=50,
        num_of_reviews = 5, 
        intra_method='dualFC', inter_method='dualFC', 
        learning_rate=0.00001, dropout=0):

        self.device = device
        self.net_type = net_type
        self.save_dir = save_dir
        
        self.voc = voc
        self.prerocess = prerocess              # prerocess method
        self.training_epoch = training_epoch
        self.latent_k = latent_k
        self.hidden_size = hidden_size
        self.num_of_reviews = num_of_reviews
        self.clip = clip

        self.intra_method = intra_method
        self.inter_method = inter_method
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.batch_size = batch_size
    
        pass

    def _get_asin_reviewer(self, select_table='clothing_'):
        """Get asin and reviewerID from file"""
        asin, reviewerID = self.prerocess._read_asin_reviewer(table=select_table)
        return asin, reviewerID

    def set_training_batches(self, training_batches, external_memorys, candidate_items, candidate_users, training_batch_labels):
        self.training_batches = training_batches
        self.external_memorys = external_memorys
        self.candidate_items = candidate_items
        self.candidate_users = candidate_users
        self.training_batch_labels = training_batch_labels
        pass
    
    def set_asin2title(self, asin2title):
        self.asin2title = asin2title
        pass