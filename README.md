

# Attention Mechanism — In-Depth Architecture & Conceptual Notes

---

##  Why Attention Was Introduced?

In traditional **Encoder–Decoder (Seq2Seq)** architecture:

Encoder → Final Hidden State (Context Vector) → Decoder

Problem:
- Entire input sequence compressed into a single fixed-length vector.
- Information bottleneck.
- Performance degrades for long sequences.
- Decoder cannot selectively focus on specific words.

This becomes a major limitation in machine translation and long-text generation.

---

##  Core Idea of Attention

Instead of using only the final encoder hidden state, we:

✔ Keep ALL encoder hidden states  
✔ Let decoder dynamically focus on relevant parts  
✔ Compute a new context vector at every decoding step  

New Flow:

Encoder → {h1, h2, h3, ..., hn}  
Decoder Step t → Attention over all hi → Context Vector Ct → Output yt

---

## Mathematical Intuition

At decoder step t:

Let:
- Encoder hidden states = h1, h2, ..., hn
- Decoder hidden state at time t = st

### Step 1: Compute Alignment Scores

We measure how relevant each encoder hidden state hi is to current decoder state st.

Score function:

et,i = score(st, hi)

Different scoring functions:

1. Dot Product:
   score(st, hi) = st^T hi

2. General:
   score(st, hi) = st^T W hi

3. Additive (Bahdanau):
   score(st, hi) = v^T tanh(W1 hi + W2 st)

---

### Step 2: Convert Scores to Probabilities

Apply Softmax:

αt,i = exp(et,i) / Σ exp(et,j)

These α values are attention weights.

Properties:
- Sum of all αt,i = 1
- Represent importance of each input token

---

### Step 3: Compute Context Vector

Weighted sum:

Ct = Σ (αt,i * hi)

This context vector captures relevant information for current decoding step.

---

### Step 4: Generate Output

Decoder uses:

yt = f(st, Ct)

Where:
- st = current decoder hidden state
- Ct = attention context vector
- yt = predicted output

---

## Types of Attention

### 1. Bahdanau Attention (Additive Attention)

- Introduced in 2015.
- Uses feed-forward neural network to compute scores.
- Works well for smaller hidden dimensions.
- More computationally expensive than dot product.

---

###  2. Luong Attention (Multiplicative / Dot-Product)

- Uses dot-product between st and hi.
- Faster and more efficient.
- Works better when hidden dimensions are large.

---

###  3. Self-Attention (Used in Transformers)

Instead of encoder-decoder interaction:

Each token attends to every other token in the SAME sequence.

Key Idea:
Query (Q), Key (K), Value (V)

Attention(Q, K, V) = softmax( (QK^T) / √dk ) V

Where:
- Q = Query matrix
- K = Key matrix
- V = Value matrix
- dk = dimension of keys (used for scaling)

---

##  Scaled Dot-Product Attention

Formula:

Attention(Q, K, V) = softmax(QK^T / √dk) V

Why divide by √dk?

Because:
- Large dot-product values push softmax into very small gradients.
- Scaling stabilizes training.

---

## Multi-Head Attention

Instead of single attention:

MultiHead(Q, K, V) = Concat(head1, head2, ..., headh) W^O

Each head:
- Learns different relationships.
- Captures different contextual patterns.
- Improves model expressiveness.

---

##  Complete Transformer Attention Architecture

1. Input Embedding  
2. Positional Encoding  
3. Multi-Head Self-Attention  
4. Add & Layer Norm  
5. Feed Forward Network  
6. Add & Layer Norm  

Encoder Block:
- Self Attention
- Feed Forward

Decoder Block:
- Masked Self Attention
- Encoder-Decoder Attention
- Feed Forward

---

##  Why Attention Is Powerful?

✔ Removes information bottleneck  
✔ Handles long sequences better  
✔ Enables parallel computation (in Transformers)  
✔ Improves translation quality  
✔ Forms backbone of GPT, BERT, LLaMA, etc.  

---

##  Limitations

- Quadratic complexity O(n²) for self-attention.
- Memory intensive for long sequences.
- Requires large compute.

---

## Summary

Attention allows the model to:
- Dynamically focus on important input tokens
- Compute context vectors at each decoding step
- Improve long-sequence learning
- Enable modern LLM architectures

Without Attention → No Transformers  
Without Transformers → No GPT / BERT  

Attention is the foundation of modern Generative AI.

---
