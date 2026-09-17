Feature: Generation orchestration - KV-cache, logits processing, sampling, prompt assembly, and streamed decoding
  InferenceEngine.Engine owns the generation loop: the KV-cache, a deterministic logits-processor
  phase (hard constraints such as banning) that runs before the probabilistic sampling pipeline,
  the chat prompt template, and turning generated token ids into correctly-decoded streamed text.

  Scenario: The KV-cache keeps every layer and position independent
    Given a KV-cache with more than one layer and more than one position
    When a key and a value are written to one (layer, position) slot
    Then reading that slot returns exactly what was written
    And no other (layer, position) slot is affected

  Scenario: Greedy decoding always returns the single most probable token
    Given a temperature of zero and a set of logits with one clear maximum
    When a token is sampled
    Then the token with the highest logit is returned

  Scenario: Top-k narrows the candidate pool to exactly k tokens
    Given a top-k value of k and a set of logits
    When top-k filtering is applied
    Then exactly the k highest logits remain eligible and every other logit is excluded

  Scenario: Top-p keeps the smallest set of tokens reaching the probability threshold
    Given a nucleus probability p and a probability distribution over tokens
    When top-p filtering is applied
    Then the smallest prefix of tokens, sorted by probability descending, whose cumulative
      probability reaches p remains eligible, and the rest are excluded

  Scenario: Sampling is reproducible given the same seed
    Given a fixed random seed and a fixed set of logits
    When sampling runs twice with that seed
    Then both runs produce the same sampled token

  Scenario: Sampling never mutates the caller's logits
    Given a set of logits passed into the sampler
    When a token is sampled
    Then the original logits values are unchanged afterward

  Scenario: Banning a token sequence excludes it even under greedy decoding
    Given a temperature of zero and a logits processor banning the single highest-logit token
    When a token is sampled
    Then the banned token is never returned, and the next-highest surviving token is returned instead

  Scenario: A banned multi-token sequence is only masked once its prefix has actually been generated
    Given a banned sequence of more than one token
    When a token is sampled after only some, not all, of that sequence's prefix has been generated
    Then the sequence's completing token is not excluded
    When a token is sampled after the full prefix has been generated in order
    Then the sequence's completing token is excluded

  Scenario: The EOS token can never be banned
    Given a banned sequence that names the EOS token, either directly or as its completing token
    When that sequence's ban would otherwise apply
    Then the EOS token's logit is left unchanged

  Scenario: Generation falls back to EOS when every other token is masked
    Given every non-EOS token excluded by logits processors
    When a token is sampled, under greedy decoding or under stochastic sampling
    Then the EOS token is returned instead of an arbitrary index

  Scenario: The chat prompt is assembled in system, user, assistant-priming order
    Given a system prompt and a user prompt
    When the chat prompt template builds the token sequence
    Then it begins with the system turn wrapped in the model's special tokens,
      followed by the user turn wrapped the same way,
      followed by the assistant turn opener with no closing token

  Scenario: A missing special token fails prompt assembly clearly
    Given a tokenizer whose vocabulary does not include the model's turn-marker special token
    When the chat prompt template tries to build a prompt
    Then a KeyNotFoundException is thrown instead of producing a malformed prompt

  Scenario: A character split across two generated tokens decodes correctly
    Given a multi-byte UTF-8 character whose bytes are split across two consecutive tokens
    When each token's bytes are streamed through the incremental decoder in order
    Then the first token yields no text and the second token yields the complete character

  Scenario: Generation rejects a negative token budget immediately
    Given a GenerationOptions with a negative MaxNewTokens
    When Generate is called
    Then an ArgumentException is thrown before any model or cache work begins,
      without needing to enumerate the result

  Scenario: Generation rejects a prompt that encodes to nothing
    Given raw mode and an empty prompt string
    When Generate is called
    Then an ArgumentException is thrown immediately

  Scenario: Generation rejects a sequence longer than the model's context
    Given a GenerationOptions whose MaxNewTokens, added to the prompt length, exceeds the
      model's maximum sequence length
    When Generate is called
    Then an ArgumentException is thrown immediately
