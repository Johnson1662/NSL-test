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
    


def attention(q, k, v, mask):  # [n_q, d_k], [n_k, d_k], [n_k, d_v], [n_q, n_k] -> [n_q, d_v]
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
    d_k = k.size(-1)
    attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)  # [n_q, d_k] @ [d_k, n_k] -> [n_q, n_k]
    attn_scores = attn_scores.masked_fill(~(mask == 0), float('-inf'))
    attn_weights = softmax(attn_scores)  # [n_q, n_k]
    return attn_weights @ v  # [n_q, n_k] @ [n_k, d_v] -> [n_q, d_v]

# Multi-Head Attention
def mha(x, attn, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd]
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
    causal_mask = torch.tril(torch.ones(x.size(0), x.size(0)))  # [n_seq, n_seq]
    causal_mask = causal_mask.masked_fill(causal_mask == 0, float('-inf')).masked_fill(causal_mask == 1, float(0.0))

    # Perform attention over each head
    out_heads = [attention(q, k, v, causal_mask) for q, k, v in qkv_heads]  # n_head * [n_seq, n_embd/n_head]
    
    # Merge heads
    """
        Task: merge multi-heads results
        Notes: n_head * [n_seq, n_embd/n_head] --> [n_seq, n_embd]
    """
    x = torch.cat(out_heads, dim=-1)  # n_head * [n_seq, n_embd/n_head] --> [n_seq, n_embd]
    
    # Out projection
    x = linear(x, c_proj)  # [n_seq, n_embd] -> [n_seq, n_embd]
    
    return x


def transformer_block(x, block, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd]
    mlp, attn, ln_1, ln_2 = block['mlp'], block['attn'], block['ln_1'], block['ln_2']
    
    # multi-head causal self attention
    x = x + mha(layer_norm(x, ln_1), attn, n_head=n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # position-wise feed forward network
    x = x + ffn(layer_norm(x, ln_2), mlp)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x


def gpt2(inputs, params, n_head):  # [n_seq] -> [n_seq, n_vocab]
    wte, wpe, blocks, ln_f = params['wte'], params['wpe'], params['blocks'], params['ln_f']
    # token + positional embeddings
    x = wte[inputs] + wpe[range(len(inputs))]  # [n_seq] -> [n_seq, n_embd]
    
    x = torch.Tensor(x)
    # forward pass through n_layer transformer blocks
    for block in blocks:
        x = transformer_block(x, block, n_head=n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # projection to vocab
    x = layer_norm(x, ln_f)  # [n_seq, n_embd] -> [n_seq, n_embd]
    return x @ wte.T  # [n_seq, n_embd] -> [n_seq, n_vocab]


def generate(inputs, params, n_head, n_tokens_to_generate):
    from tqdm import tqdm

    for _ in tqdm(range(n_tokens_to_generate), "generating"):  # auto-regressive decode loop
        logits = gpt2(inputs, params, n_head=n_head)  # model forward pass
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

    from tqdm import tqdm

    generated_ids = []
    current_inputs = list(inputs)

    with tqdm(total=n_tokens_to_generate, desc="Speculative Generating") as progress_bar:
        while len(generated_ids) < n_tokens_to_generate:
            # Draft model generates K tokens
            draft_ids = []
            draft_inputs = list(current_inputs)
            for _ in range(K):
                draft_logits = gpt2(draft_inputs, draft_params, n_head=hparams_draft["n_head"])
                next_id = int(np.argmax(draft_logits[-1]))
                draft_ids.append(next_id)
                draft_inputs.append(next_id)

            # Target model verifies all K tokens in a single forward pass
            target_logits = gpt2(draft_inputs, target_params, n_head=hparams_target["n_head"])
            verification_logits = target_logits[len(current_inputs)-1:-1]

            # Greedy Sampling by target model
            accepted_counts = 0
            for i in range(K):
                target_next_id = int(np.argmax(verification_logits[i]))
                if target_next_id == draft_ids[i]:
                    accepted_counts += 1
                else:
                    # correct the first wrong token
                    extra_token = target_next_id 
                    break
            
            if accepted_counts == K:
                extra_token = int(np.argmax(target_logits[-1]))

            # Add accepted tokens to the final output
            accepted_ids = draft_ids[:accepted_counts] + [extra_token]
            
            for token_id in accepted_ids:
                if len(generated_ids) < n_tokens_to_generate:
                    generated_ids.append(token_id)
                    current_inputs.append(token_id)
                    progress_bar.update(1)
                else:
                    break
            
            if len(generated_ids) >= n_tokens_to_generate:
                break

    return generated_ids


def main(prompt: str, n_tokens_to_generate: int = 5, draft_model_size: str = "124M", target_model_size: str = "1558M", models_dir: str = "models"):
    from utils import load_encoder_hparams_and_params

    # load encoder, hparams, and params from the released open-ai gpt-2 files
    draft_encoder, draft_hparams, draft_params = load_encoder_hparams_and_params(draft_model_size, models_dir)
    target_encoder, target_hparams, target_params = load_encoder_hparams_and_params(target_model_size, models_dir)

    # encode the input string using the BPE tokenizer
    draft_input_ids = draft_encoder.encode(prompt)
    target_input_ids = target_encoder.encode(prompt)

    # make sure we are not surpassing the max sequence length of our model
    assert len(draft_input_ids) + n_tokens_to_generate < draft_hparams["n_ctx"]
    assert len(target_input_ids) + n_tokens_to_generate < target_hparams["n_ctx"]

    # generate output ids
    start = time.time()
    # output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)
    output_ids = greedy_speculative_generate(draft_input_ids, draft_params, target_params, draft_hparams, target_hparams, n_tokens_to_generate, K=4)
    end = time.time()
    print(f"Time taken to generate {n_tokens_to_generate} tokens: {end - start:.2f}s")

    # decode the ids back into a string
    output_text = draft_encoder.decode(output_ids)
    return output_text


if __name__ == "__main__":
    import fire
    fire.Fire(main)