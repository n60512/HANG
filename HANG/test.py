import torch

a = torch.randn(5, 80, 300)
b = torch.randn(80, 300)

c = a*b

d = torch.sum(a * b, dim=2)

e = d.unsqueeze(2)
stop = 1

# energy = self.attn(query_vector)
# x = F.relu(key_vector * energy)
# weighting_score = self.linear_beta(x)
# # Calculate attention score            
# inter_attn_score = torch.softmax(weighting_score, dim = 0)