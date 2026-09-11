Feature: Byte-level BPE tokenization matches the model's trained vocabulary
  InferenceEngine.Tokenizers hand-rolls a byte-level BPE tokenizer (the GPT-2 family) built
  directly from a model's embedded vocabulary and merge ranks, since GGUF embeds the vocabulary
  but not the pre-tokenizer's splitting regex.

  Scenario: Text survives an encode/decode round trip
    Given a tokenizer built from a small vocabulary and merge list
    When a piece of text is encoded and then decoded
    Then the decoded text equals the original text

  Scenario: The lowest-rank merge is always applied, not the first pair scanned
    Given a tokenizer whose merge list ranks "b c" ahead of "a b"
    And the symbols "a", "b", "c" adjacent to each other
    When BPE merging runs
    Then "b" and "c" are merged first, even though "a" and "b" is the first pair the scan reaches

  Scenario: Digits are pre-tokenized individually
    Given a tokenizer using the "smollm" pre-tokenizer variant
    When text containing letters, then a digit, then more letters is encoded
    Then the digit becomes its own separate token, distinct from the letters around it

  Scenario: A special token is found by its literal text
    Given a tokenizer whose vocabulary includes a literal control token
    When that token's exact text is looked up
    Then its token id is returned

  Scenario: An unrecognized pre-tokenizer variant fails at construction
    Given a TokenizerData whose PreTokenizerName is not "smollm"
    When a tokenizer is created from it
    Then a NotSupportedException is thrown immediately, before any text is processed

  Scenario: Every byte round-trips through the byte-level alphabet
    Given each possible byte value from 0 to 255
    When the byte is mapped to its printable character and back to a byte
    Then the original byte value is recovered
