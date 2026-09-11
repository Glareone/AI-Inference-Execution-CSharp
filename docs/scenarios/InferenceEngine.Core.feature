Feature: Core data contracts have value semantics
  InferenceEngine.Core holds the shared data contracts (ModelConfig, TokenizerData) and
  interfaces (IModel, ITokenizer, IKvCache) that every other layer depends on. It has no logic
  of its own — the only thing worth specifying is that its record types behave like values,
  since a future caller (e.g. a model cache) may key off equality.

  Scenario: Two ModelConfig instances describing the same model are equal
    Given a ModelConfig built from a set of architecture values
    And a second ModelConfig built from the exact same values
    When the two instances are compared
    Then they are equal
    And their hash codes are equal

  Scenario: A ModelConfig with one different field is not equal
    Given a ModelConfig built from a set of architecture values
    And a copy of it with one field changed
    When the two instances are compared
    Then they are not equal

  Scenario: TokenizerData built from the same array instances is equal
    Given a TokenizerData built from a Tokens array and a Merges array
    And a second TokenizerData built from those exact same array instances
    When the two instances are compared
    Then they are equal

  Scenario: TokenizerData built from separately-parsed but identical arrays is not equal
    Given a TokenizerData built from a Tokens array and a Merges array
    And a second TokenizerData built from newly-allocated arrays with the same content
    When the two instances are compared
    Then they are not equal
    And this is a documented production-code gap, not a passing guarantee — record equality
      compares string[] fields by reference, not by content
