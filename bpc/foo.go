package bpc

import ort "github.com/yalue/onnxruntime_go"

func generate(input string, maxNewBytes int) {
	// input: unprocessed string; must end with valid utf8
	// maxNewBytes: number of bytes to generate

	// TODO handle byte replacements
	// TODO convert into list of bytes (validate against ref)
	// TODO track sampler state
	// TODO track last byte

	for range maxNewBytes {
		// TODO call probNectByte
		// TODO just select best for now
		// TODO append best to input bytes
		// TODO set last byte and sampler state
	}
}

type samplerState struct {
	P_tT           *ort.Tensor[float64]
	mask           *ort.Tensor[float64]
	utf8Seq        string // string -> []byte -> []int8
	cvrsLogprobs   *ort.Tensor[float64]
	covers         []string
	coverEncs      [][]int64
	prevContextEnc []int64
	divEnc         []int64
}

func probNextByte(raw string, state samplerState, lastByte byte) {
	// BytePredLLM

	// raw: raw_utf8_seq prompt as byte input (x^n_1)
	// sampler: meta data from previous runs: cover(x^{n-1}|x^n_1)

	// TODO handle prefix space
	// TODO step 1: obtain cover(x^n_1)
	//      if state is empty run extractCoverProbBytes
	//      else update sampler state
	// TODO step 2: P(x|T) T = enc(utf8_seq); P(x^{n+1}|x_{n}_1)
	//      run nextByteUpdate
	// TODO return predictions and sampler state
}

func updateSamplerState() {
	// BytePredLLM

	// Extract the cover encodings of the input utf8-seq, i.e. obtain cover(x^n_1) from cover(x^{n-1}_1).

	// TODO step 1
	//      load current state
	//      encode new context
	// TODO step 2: extract cover encoding of x^{x+1}_1
	//      a) P(t, T): filter t were x_{n+1} + t_1 ?1
	//      b) Remove t = cover(x^n_1) and d nt contain x_{n+1} ?!

	// TODO encode context
	// TODO log_PtT = torch.log(P+tT)
	// TODO find token hat start with last byte; validate
	// TODO remove last byte from covers
	// TODO update converged encodings

	// Returns:
	// cvrs_logprobs: log probability of cover of x^n_1
	// ctx_enc: encode of (x^n_1)
	// covers: cover string of x^n_1
	// new_cover_encs: cover encodings of x^n_1.
	// utf8_seq: x^n_1
	// new_div_enc: Encoding that stops at the last white space in x^n_1.
}

func nextByteUpdate() {
	// BytePredLLM

	// Compute P(x_{n+1}|x^n_1).
	// Step 1: Compute logP(t|T), T = ctx_enc, also rescale.
	// Step 2: Compute P(x, T) through marginalization.
	// Step 3: Compute P(x, T') for other cover encodings T'.
	// Step 4: Compute P(x|prev_x).
	// Conditioning separated by _, e.g. log_pt_T = log(p_t|T)

	// Args:
	// cvrs_logprobs: log probability of cover of x^n_1
	// ctx_enc: encode of (x^n_1)
	// covers: cover string of x^n_1
	// cover_encs: cover encodings of x^n_1.
	// utf8_seq: x^n_1
	// fast_mode (bool): False for disable truncation.
	// div_enc (List[Int]): Encoding that stops at the last white space in x^n_1
	// sampler_state: contains cover(x^{n-1}_1) and KV caches.

	// TODO step 1
	//      call runKV
	//      rescale logprobs
	// TODO step 2
	//      log_pT = cvrs_logprobs[cover_encs.index(ctx_enc[len(div_enc) :])]
	//      ptT = torch.exp(log_pT + log_pt_T) * mask
	//      pxT = torch.matmul(self.t_wrapper.sum_prefix_idx, ptT)
	// TODO step 3
	//      other cover encodings T' ?!
	// TODO step 4
	//      P_unorm = torch.matmul(mat_sum.to(self.device), cvrs_scaled_probs) + pxT
	//      predictions = P_unorm / P_unorm.sum()
	// TODO update kv cache
	// TODO return prediction
	// TODO return new kv cache

	// Returns:
	// predictions: P(x_{n+1}|x^n_1)
	// sampler_state: new meta data, contains cover(x^n_1).
}

func runKV(ctxEnc []int64, state *samplerState, divEnc []int64) {
	// BytePredLLM

	// log P(t_{i+1}| t^i_1)

	// TODO if no sampler state init cache
	// TODO runTR with cache
	// TODO (fast mode)

	// Returns:
	// log_pt_T (torch.Tensor): log P(t_{i+1}| t^i_1) for all t_{i+1}
	// mask (torch.Tensor): invalid encoding masks.
	// cache : KV cache of t^i.
}

func extractCoverProbBytes(raw string) {
	// BytePredLLM

	// TODO
	// Note:
	// Step 1: For simplification, we force the string to end with a valid utf-8 character.
	// It must also contain a white space (exclude the dummy prefix one as in llama2)
	// since some models such as Yi does not provide P(t_1).

	// TODO step 1: preprocess
	//      handle byte replacements
	//      whitespace split into query and condition
	//      encode condition
	//      str2bytes ?! for query
	// TODO step 2: extract cover encodings and compute prob
	//      P(t | div_enc)
	//      build proposal encodings
	//      validate proposal encodings
	//      run coverTokenLikelihoods
	// TODO return cover strings, cover logprobs, covers enc and div enc
}

func coverTokenLikelihoods(left, diverged []int64) {
	// BytePredLLM

	// Compute logP(t_{m+1}=t, t^m_n|t^n_1). See diagram.
	// |  t^n_1       |       t^m_n      |   t_{m+1}    |
	// |  div_enc     |    left tokens   | right tokens |
	// |          left_enc               | right tokens |

	// TODO copy div tokens and use as context
	// TODO runTR for left enc (no cache)
	// TODO runTR for each quert token; append to context (no cache)
	// TODO likelihood per encs = logpval + logprobs ?!

	// torch.Tensor: logP(t_{m+1}=t, t^m_n|t^n_1)
	// mask (torch.Tensor): invalid map (1.0 = valid, 0.0 = invalid)
}

func runTR() {
	// BytePredLLM

	// TR ?!

	// TODO (KV cache)
	// TODO call logprobsNextToken
	// TODO validate results
	// TODO (fast mode no validation)
}

func logprobsNextToken(inputIDs []int64) {
	// base model

	// TODO (KV cache)
	// TODO self.llm
	// TODO softmax
	// TODO return log probs
}
