import torch
import math
import numpy as np

def scaled_dot_product_attention_solution(Q, K, V):
    #TODO
    d_k = Q.shape[-1]
    return torch.matmul(torch.softmax(torch.matmul(Q, torch.transpose(K, 1, 2))/np.sqrt(d_k), dim=-1), V)


class MultiHeadAttentionSolution:
    def __init__(self, d_model, num_heads):
        self.W_q = torch.nn.Linear(d_model, d_model)
        self.W_k = torch.nn.Linear(d_model, d_model)
        self.W_v = torch.nn.Linear(d_model, d_model)
        self.W_o = torch.nn.Linear(d_model, d_model)
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

    def forward(self, Q, K, V):
        q = self.W_q(Q) #[batch, n_seq, d_model]
        k = self.W_k(K)
        v = self.W_v(V)

        n_batch, q_n_seq, dim = Q.shape
        k_n_seq = K.shape[1]
        
        q = q.reshape(n_batch, q_n_seq, -1, self.d_k).transpose(1,2)
        k = k.reshape(n_batch, k_n_seq, -1, self.d_k).transpose(1,2)
        v = k.reshape(n_batch, k_n_seq, -1, self.d_k).transpose(1,2)

        attn = torch.matmul(q, k.transpose(2,3))/math.sqrt(self.d_k)
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1,2).reshape(n_batch, q_n_seq, self.d_model)
        out = self.W_o(out)
        return out