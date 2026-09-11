Feature: Loading a GGUF model file and running its transformer math
  InferenceEngine.Models owns model loading (parsing the GGUF binary format, memory-mapping
  weights) and the Llama forward pass, built on System.Numerics.Tensors with no custom kernels.
  Tests exercise the reader against small synthetic GGUF files, not the real 258 MB model.

  Scenario: Scalar metadata is read back correctly
    Given a synthetic GGUF file with a string, a u32, and an f32 metadata entry
    When the file is opened
    Then each value is returned with its original type and content

  Scenario: Array metadata is read back correctly
    Given a synthetic GGUF file with a string-array metadata entry
    When the file is opened
    Then the array is returned with its original elements in order

  Scenario: A missing optional metadata key falls back to its default
    Given a synthetic GGUF file without a given optional key
    When that key is read with a caller-supplied default
    Then the default value is returned

  Scenario: An F32 tensor round-trips exactly
    Given a synthetic GGUF file containing an F32 tensor
    When the tensor is read
    Then the returned values equal the original floats exactly

  Scenario: An F16 tensor is dequantized to F32
    Given a synthetic GGUF file containing an F16 tensor
    When the tensor is read
    Then the returned values are the F32 equivalents of the original half-precision floats

  Scenario: An unsupported quantized tensor type fails loudly
    Given a synthetic GGUF file containing a Q4_0-quantized tensor
    When the tensor is read
    Then a NotSupportedException is thrown instead of silently returning wrong data

  Scenario: Multiple tensors' byte offsets never overlap
    Given a synthetic GGUF file containing two or more tensors after the aligned data section
    When each tensor is read
    Then each tensor's values are correct and independent, with no gap or overlap between them

  Scenario: A custom alignment value is honored
    Given a synthetic GGUF file whose general.alignment metadata overrides the 32-byte default
    When the file is opened and a tensor is read
    Then the tensor data is located using the custom alignment, not the default

  Scenario: RMSNorm matches its mathematical definition
    Given an input vector, a weight vector, and an epsilon
    When RMSNorm is applied
    Then the result equals the input scaled by 1/sqrt(mean(x^2) + eps) and the weight

  Scenario: RoPE is a no-op at position zero
    Given any query or key vector and any rope frequency base
    When rotary position embedding is applied at position 0
    Then the vector is unchanged

  Scenario: Softmax produces a valid probability distribution
    Given a vector of logits
    When softmax is applied
    Then every output is non-negative and all outputs sum to 1

  Scenario: MatVec matches manual dot products
    Given a small row-major [out, in] weight matrix and an input vector
    When MatVec is applied
    Then each output element equals the dot product of its row with the input

  Scenario: SwiGLU is zero when the gate is zero
    Given a gate vector of all zeros and an arbitrary up vector
    When SwiGLU is applied
    Then the result is all zeros
