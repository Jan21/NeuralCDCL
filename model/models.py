"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn_values = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_attn_keys = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_attn_queries = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        #self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        #if not self.flash:
        #    print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
        bias = 1 - torch.eye(config.block_size, config.block_size).view(1, 1, config.block_size, config.block_size)
        self.register_buffer("bias", bias)
        # flip 0s and 1s


    def forward(self, x, keys, queries, mask):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        C2 = keys.shape[-1]
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        #v  =  self.c_attn_values(x) #self.c_attn(x).split(self.n_embd, dim=2)
        q = queries
        k = keys
        v = x
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        mask = mask.unsqueeze(1).unsqueeze(-1)
        v = v * mask
        k = k * mask
        q = q * mask

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        #if self.flash:
            # efficient attention using Flash Attention CUDA kernels
        #    y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        #else:
            # manual implementation of attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        #att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        #att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.n_embd)
        # attn_weights = F.softmax(attn_scores, dim=-1)
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y, att

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, keys, queries, mask):
        x = x + self.attn(self.ln_1(x),keys,queries,mask)[0]
        # x = x + self.attn(x)[0]
        #x = x + self.mlp(self.ln_2(x))
        return x
 # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.num_vars= config.num_vars
        self.num_out = config.num_classes
        self.n_embd = config.n_embd
        self.n_iters = config.n_layer
        self.clause_updater = nn.LSTM(config.n_embd, config.n_embd)
        self.lit_updater = nn.LSTM(2*config.n_embd, config.n_embd)
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wte2 = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config)]),# for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))

        self.node_embed = nn.Linear(config.n_embd, config.n_embd)
        self.pos_embed = ScalarEmbeddingSine1D(config.n_embd, normalize=False)
 
        self.time_embed = nn.Sequential(
            nn.Linear( config.n_embd, config.n_embd),
            nn.ReLU(),
            nn.Linear(config.n_embd, config.n_embd ),
        )

        self.lm_head = nn.Linear(config.n_embd, config.num_classes, bias=False)
        #self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def timestep_embedding(self, timesteps, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
    
        :param timesteps: a 1-D Tensor of N indices, one per batch element.
                        These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an [N x dim] Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def random_init_embeddings(self, input_ids, xt,device):
        total_tokens = input_ids.shape[1]
        b = input_ids.shape[0]
        # Flip the value of xt from 0 to 1 or 1 to 0
        not_xt = 1 - xt
        num_lits = xt.shape[-1]*2
        num_clauses = total_tokens - num_lits
        xt = torch.cat([xt, not_xt], dim=-1)
        xt = xt.to(device)
        # pos_embed will lift the scalars for variables into a high-dim vector space, node_embed is just a linear layer
        # TODO try to eliminate the pos_embed and use only node_embeding which will do the lifting.
        x_unk = self.node_embed(self.pos_embed(xt))
        x_unk = x_unk / torch.norm(x_unk, dim=1, keepdim=True)
        # init clause states
        x_c = torch.zeros((b,num_clauses,self.n_embd), requires_grad=False).to(device)
        # init cell states
        x_unk_h = torch.zeros(x_unk.shape).to(device)
        x_c_h = torch.zeros(x_c.shape).to(device) 
        x_tok = torch.cat([x_unk, x_c], dim=-2)
        h_tok = torch.cat([x_unk_h, x_c_h], dim=-2)     
        return x_tok, h_tok


    def forward(self, inputs, xt,timestep):

        input_ids, pos_embeddings, masks, labels, att_mask = inputs.values()
        device = input_ids.device
        b, t = input_ids.size()

        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        # pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
        time_emb = self.time_embed(self.timestep_embedding(timestep.to(device), self.n_embd)).unsqueeze(1)
        # forward the GPT model itself
        tok_emb = self.transformer.wte(input_ids) # token embeddings of shape (b, t, n_embd)
        value_embs,value_h = self.random_init_embeddings(input_ids, xt,device)

        # reinitialize the last token to random unit vector
        # tok_emb[:,-1,:] = torch.nn.init.normal_(torch.empty(tok_emb[:,-1,:].shape), mean=0.0, std=0.02).to(device)
        pos_emb = self.transformer.wte(pos_embeddings) # position embeddings of shape (t, n_embd)
        # pos bude 3D tensor batch * k * len(sequence)
        # navíc bude feature dimenze
        # zamaskují se feature vektory
        pos_emb_masked = torch.sum(pos_emb * masks.unsqueeze(-1), axis=1)
        # sečtu - pro každý token sečtu vektory
        # pos_emb už bude sečtený
        queries = tok_emb
        keys = pos_emb_masked
        x = value_embs #self.transformer.drop(tok_emb + pos_emb_masked + value_embs)
        #x = self.transformer.h[0](x) + time_emb
        block = self.transformer.h[0]
        value_embs_pos_lits = value_embs[:,:self.num_vars, :].reshape(1,-1,self.n_embd)
        value_h_pos_lits = value_h[:,:self.num_vars, :].reshape(1,-1,self.n_embd)
        value_embs_neg_lits = value_embs[:,self.num_vars:2*self.num_vars, :].reshape(1,-1,self.n_embd)
        value_h_neg_lits = value_h[:,self.num_vars:2*self.num_vars, :].reshape(1,-1,self.n_embd)
        value_embs_lits = torch.cat([value_embs_pos_lits, value_embs_neg_lits], dim=1)
        value_h_lits = torch.cat([value_h_pos_lits, value_h_neg_lits], dim=1)
        value_embs_clauses = value_embs[:,2*self.num_vars:, :].reshape(1,-1,self.n_embd)
        value_h_clauses = value_h[:,2*self.num_vars:, :].reshape(1,-1,self.n_embd)
        hidden_lits = (value_embs_lits, value_h_lits)
        hidden_clauses = (value_embs_clauses, value_h_clauses)
        for i in range(self.n_iters): # experiment
            x = block(x,keys,queries,att_mask) + time_emb
            # Split x into three parts along sequence dimension
            x_pos = x[:, :self.num_vars, :].reshape(1,-1,self.n_embd)  # Pos lits tokens tokens
            x_neg = x[:, self.num_vars:2*self.num_vars, :].reshape(1,-1,self.n_embd)  # Neg lits tokens  
            x_clauses = x[:, 2*self.num_vars:, :].reshape(1,-1,self.n_embd) # clause tokens
            pos_neg = torch.cat([x_pos, x_neg], dim=-1)
            neg_pos = torch.cat([x_neg, x_pos], dim=-1)
            x_lits = torch.cat([pos_neg, neg_pos], dim=1)
            # Recombine in same order
            #x = torch.cat([x_pos, x_neg, x_clauses], dim=1)
            msg, hidden_clauses = self.clause_updater(x_clauses, hidden_clauses)
            msg, hidden_lits = self.lit_updater(x_lits, hidden_lits)
            x_clauses = hidden_clauses[0].reshape(b,-1,self.n_embd)
            x_pos_lits, x_neg_lits = torch.chunk(hidden_lits[0], 2, dim=1)
            x_pos_lits = x_pos_lits.reshape(b,-1,self.n_embd)
            x_neg_lits = x_neg_lits.reshape(b,-1,self.n_embd)
            x = torch.cat([x_pos_lits, x_neg_lits, x_clauses], dim=1)
            #            unk_hidden += time_emb 
            #hidden0 = hidden[0] / torch.norm(hidden[0], dim=1, keepdim=True)  
            #hidden = (hidden0, hidden[1])
            #x = hidden[0].reshape(b,-1,self.n_embd)
        x = self.transformer.ln_f(x)

        if input_ids is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            # Reshape logits to match labels dimensions (batch, classes, sequence)
            return logits
            # Binary cross entropy with logits (includes sigmoid)
            #loss = F.binary_cross_entropy_with_logits(logits[:,:,:self.num_vars], labels[:,:,:self.num_vars].float())
            #loss = F.cross_entropy(logits.view(-1, logits.size(-1)), input_ids.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None
        return logits, loss


    def configure_optimizers(self, weight_decay, learning_rate, betas):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)

        return optimizer


class ScalarEmbeddingSine1D(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
    
    # def forward(self, x):
    #     x_embed = x
    #     dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
    #     dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)
    
    #     pos_x = x_embed[:, None] / dim_t
    #     pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
    #     return pos_x
    
    def forward(self, x):
        # x shape: (batch_size, seq_length)
        x_embed = x
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)
        
        # Add broadcasting dimension for dim_t: (1, 1, num_pos_feats)
        dim_t = dim_t.view(1, 1, -1)
        
        # Add feature dimension to x_embed: (batch_size, seq_length, 1)
        pos_x = x_embed.unsqueeze(-1) / dim_t
        
        # Stack sin and cos: (batch_size, seq_length, num_pos_feats)
        pos_x = torch.stack(
            (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()),
            dim=-1
        ).flatten(-2)  # Flatten last 2 dimensions
        
        return pos_x