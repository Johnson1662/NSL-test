import numpy as np
import torch
import time
import math
torch.set_printoptions(8)

def gelu(x):
    """
        Task: Use the torch API to implement the approximate calculation formula of the `GELU`
        activation function. The formula is as follows (you need to paste it into the latex
        online conversion website)
        Website: https://www.latexlive.com/
        Formula: \frac{1}{2} x\left[1+\tanh \left(\sqrt{\frac{2}{\pi}}\left(x+0.044715 x^{3}\right)\right)\right]
        
        Input: Tensor
        Output: Tensor
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


def softmax(x):
    """
        Task: Use torch API to implement `softmax` function, search the specific formula by yourself
        Input: Tensor
        Output: Tensor
    """
    x = x - torch.max(x, dim=-1, keepdim=True).values
    exp_x = torch.exp(x)
    sum = torch.sum(exp_x, dim=-1, keepdim=True)
    return exp_x / sum

def layer_norm(x, g_b, eps:float = 1e-5):
    """
        Task: Use torch API to implement `layernorm` function, search `layernorm` by yourself
        Input: 
            x: Tensor
            g_b: dictionary that load from gpt2 weight. g-gamma and b-bias are the keys
        Output: Tensor
    """
    g, b = torch.Tensor(g_b['g']), torch.Tensor(g_b['b'])
    mean = torch.mean(x, dim=-1, keepdim=True)
    std = torch.std(x, dim=-1, keepdim=True)
    eps = eps
    return g * (x - mean) / (std + eps) + b
    
def linear(x, w_b):  # [m, in], [in, out], [out] -> [m, out]
    """
        Task: implement linear layer 
        Input: 
            x: Tensor
            w_b: dictionary that load from gpt2 weight. w-weight and b-bias are the keys
        Output: Tensor
    """
    w, b = w_b['w'], w_b['b']
    return x @ torch.Tensor(w) + torch.Tensor(b)
    
# Feed Forward Network
def ffn(x, mlp):  # [n_seq, n_embd] -> [n_seq, n_embd]
    """
        Task: use `gelu` `linear` to implement ffn
        Notes: x --linear--> --gelu--> --linear--> output
        Input: 
            x: Tensor
            mlp: dictionary that load from gpt2 weight. w_b1 and w_b2 are the params of two linear layer
        Output: Tensor
    """
    w_b1, w_b2 = mlp['c_fc'], mlp['c_proj']
    x = linear(x, w_b1)
    x = gelu(x)
    x = linear(x, w_b2)
    return x
    


def attention(q, k, v, mask, head_cache):  # [n_q, d_k], [n_k, d_k], [n_k, d_v], [n_q, n_k] -> [n_q, d_v]
    """
        Task: use torch API to implement attention computation according to formula(1) of the following paper
              where d_k account for the last dimension of `k`
        Paper: https://arxiv.org/abs/1706.03762
        Input: 
            q: Tensor
            k: Tensor
            v: Tensor
            mask: Tensor
            mlp: dictionary that load from gpt2 weight. w_b1 and w_b2 are the params of two linear layer
        Output: Tensor
    """
    past_len = 0
    if head_cache["k"] is not None:
        past_len = head_cache["k"].size(0)
        k = torch.cat([head_cache["k"], k], dim=0)
        v = torch.cat([head_cache["v"], v], dim=0)

    head_cache["k"], head_cache["v"] = k, v

    d_k = k.size(-1)
    attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)  # [n_q, d_k] @ [d_k, n_k] -> [n_q, n_k]
    attn_scores = attn_scores + mask[past_len:past_len + q.size(0), :k.size(0)]
    attn_weights = softmax(attn_scores)  # [n_q, n_k]
    return attn_weights @ v  # [n_q, n_k] @ [n_k, d_v] -> [n_q, d_v]

# Multi-Head Attention
def mha(x, attn, n_head, kv_cache):  # [n_seq, n_embd] -> [n_seq, n_embd]
    """
        Task: Complete the code of the multi-head attention
        
        Input: 
            x: Tensor
            attn: dictionary that load from gpt2 weight. c_attn and c_proj are the params of two linear layer
            n_head: number of head
        Output: Tensorying multi-head attention and linear transformation, shape [n_seq, n_embd].
    """
    c_attn, c_proj = attn['c_attn'], attn['c_proj']
    # qkv projection
    x = linear(x, c_attn)  # [n_seq, n_embd] -> [n_seq, 3*n_embd]
    
    # Split into qkv
    """
        Task: Split the q,k,v matrix from the tensor x
        Notes: [n_seq, 3*n_embd] -> 3 * [n_seq, n_embd]
    """
    qkv = x.chunk(3, dim=-1)  # [n_seq, 3*n_embd] -> 3 * [n_seq, n_embd]

    # Split into heads
    qkv_heads = [qkv_part.chunk(n_head, dim=-1) for qkv_part in qkv]  # 3 * [n_seq, n_embd] -> 3 * n_head * [n_seq, n_embd/n_head]
    qkv_heads = list(zip(*qkv_heads))  # [3, n_head, n_seq, n_embd/n_head]

    # 将每个头的缓存也分割开
    if kv_cache["k"] is not None:
        cached_k_heads = kv_cache["k"].chunk(n_head, dim=-1)
        cached_v_heads = kv_cache["v"].chunk(n_head, dim=-1)
    else:
        cached_k_heads = [None] * n_head
        cached_v_heads = [None] * n_head
    
    # Causal mask to hide future inputs from being attended to
    """
        Task: Construct mask matrix
        Notes: 
            | 0  -inf -inf ... -inf |
            | 0    0  -inf ... -inf |
            | 0    0    0  ... -inf |
            |...  ...  ... ...  ... | 
            | 0    0    0  ...   0  |
        Mask is a tensor whose dimension is [n_seq, n_seq]
    """
    cur_len = x.size(0) + (cached_k_heads[0].size(0) if cached_k_heads[0] is not None else 0)
    causal_mask = torch.tril(torch.ones(cur_len, cur_len))  # [n_seq, n_seq]
    causal_mask = causal_mask.masked_fill(causal_mask == 0, float('-inf')).masked_fill(causal_mask == 1, float(0.0))

    # Perform attention over each head
    out_heads = []
    updated_head_caches = []
    for i, (q, k, v) in enumerate(qkv_heads):
        head_cache = {"k": cached_k_heads[i], "v": cached_v_heads[i]}
        out_heads.append(attention(q, k, v, causal_mask, head_cache))
        updated_head_caches.append(head_cache)
    
    kv_cache["k"] = torch.cat([hc["k"] for hc in updated_head_caches], dim=-1)
    kv_cache["v"] = torch.cat([hc["v"] for hc in updated_head_caches], dim=-1)
    
    # Merge heads
    """
        Task: merge multi-heads results
        Notes: n_head * [n_seq, n_embd/n_head] --> [n_seq, n_embd]
    """
    x = torch.cat(out_heads, dim=-1)  # n_head * [n_seq, n_embd/n_head] --> [n_seq, n_embd]
    
    # Out projection
    x = linear(x, c_proj)  # [n_seq, n_embd] -> [n_seq, n_embd]
    
    return x


def transformer_block(x, block, n_head, kv_cache):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # mlp(Feed Forward Network), attn(Multi-Head Attention)
    # ln_1(Layer Norm 1), ln_2(Layer Norm 2)
    mlp, attn, ln_1, ln_2 = block['mlp'], block['attn'], block['ln_1'], block['ln_2']
    
    # multi-head causal self attention
    x = x + mha(layer_norm(x, ln_1), attn, n_head=n_head, kv_cache=kv_cache)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # position-wise feed forward network
    x = x + ffn(layer_norm(x, ln_2), mlp)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x


def gpt2(inputs, params, n_head, layer_cache):  # [n_seq] -> [n_seq, n_vocab]
    # wte(Word Token Embeddings), wpe(Word Position Embeddings)
    # blocks(Transformer Blocks), ln_f(final Layer Norm)
    wte, wpe, blocks, ln_f = params['wte'], params['wpe'], params['blocks'], params['ln_f']
    
    # token + positional embeddings
    pos_start = layer_cache[0]["k"].size(0) if layer_cache[0]["k"] is not None else 0
    pos = range(pos_start, pos_start + len(inputs))
    x = wte[inputs] + wpe[pos]  # [n_seq] -> [n_seq, n_embd]
    
    x = torch.Tensor(x)
    # forward pass through n_layer transformer blocks
    for i, block in enumerate(blocks):
        x = transformer_block(x, block, n_head=n_head, kv_cache=layer_cache[i])  # [n_seq, n_embd] -> [n_seq, n_embd]

    # projection to vocab
    x = layer_norm(x, ln_f)  # [n_seq, n_embd] -> [n_seq, n_embd]
    return x @ wte.T  # [n_seq, n_embd] -> [n_seq, n_vocab]


def generate(inputs, params, n_head, n_tokens_to_generate):
    from tqdm import tqdm

    # initialize kv_cache for each layer
    num_layers = len(params['blocks'])
    layer_cache = [{"k": None, "v": None} for _ in range(num_layers)]
    logits = gpt2(inputs, params, n_head=n_head, layer_cache=layer_cache)  
    next_id = np.argmax(logits[-1]) 
    inputs.append(int(next_id))  
    
    for i in tqdm(range(n_tokens_to_generate - 1), "generating"):  # auto-regressive decode loop
        logits = gpt2([inputs[-1]], params, n_head=n_head, layer_cache=layer_cache)  # model forward pass
        next_id = np.argmax(logits[-1])  # greedy sampling
        inputs.append(int(next_id))  # append prediction to input

    return inputs[len(inputs) - n_tokens_to_generate :]  # only return generated ids

def greedy_speculative_generate(inputs, draft_params, target_params, hparams_draft, hparams_target, n_tokens_to_generate, K):
    
    """
        Task: Load 124M and 1558M models at the same time, use greedy sampling, and complete speculative decoding
    
        Inputs:
            inputs (list): The initial list of token IDs from the prompt.
            draft_params, target_params: Model weights for the draft and target models.
            hparams_draft, hparams_target: Hyperparameters for both models.
            n_tokens_to_generate (int): The number of new tokens to generate.
            K (int): The number of tokens the draft model speculates at each step (e.g., 4).

        Returns:
            list: A list of newly generated token IDs.
            
    """
    generated_ids = []
    current_inputs = list(inputs)

    while len(generated_ids) < n_tokens_to_generate:
        pass

    return generated_ids


def main(prompt: str, n_tokens_to_generate: int = 5, model_size: str = "124M", models_dir: str = "models"):
    from utils import load_encoder_hparams_and_params

    # load encoder, hparams, and params from the released open-ai gpt-2 files
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)

    # encode the input string using the BPE tokenizer
    input_ids = encoder.encode(prompt)

    # make sure we are not surpassing the max sequence length of our model
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]

    # generate output ids
    start = time.time()
    output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)
    end = time.time()
    print(f"Time taken to generate {n_tokens_to_generate} tokens: {end - start:.2f}s")

    # decode the ids back into a string
    output_text = encoder.decode(output_ids)
    return output_text


if __name__ == "__main__":
    import fire
    fire.Fire(main)