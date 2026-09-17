Feature: CLI configuration and error reporting
  InferenceEngine.Cli resolves configuration from a .env file, environment variables, and
  command-line flags, and reports bad input clearly instead of crashing with a raw stack trace.

  Scenario: A command-line flag overrides an environment variable
    Given INFERENCE_MODEL is set in the environment
    And "--model" is also passed on the command line
    When configuration is loaded
    Then the command-line value is used

  Scenario: An environment variable is used when no flag is given
    Given INFERENCE_MODEL is set in the environment
    And no "--model" flag is passed
    When configuration is loaded
    Then the environment variable's value is used

  Scenario: A command-line flag overrides an environment variable for banned words
    Given INFERENCE_BAN_WORDS is set in the environment
    And "--ban-words" is also passed on the command line, comma-separated and untrimmed
    When configuration is loaded
    Then the command-line value is used, split on commas with no whitespace trimmed

  Scenario: A .env file supplies a default when nothing else does
    Given a .env file setting INFERENCE_MAX_TOKENS
    And no matching environment variable or command-line flag
    When configuration is loaded
    Then the .env file's value is used

  Scenario: Command-line flags beat environment variables, which beat .env defaults
    Given a .env file, an environment variable, and a command-line flag all setting different
      values for the same option
    When configuration is loaded
    Then the command-line flag's value wins

  Scenario: Missing --model is rejected before any model loading happens
    Given no "--model" flag, no INFERENCE_MODEL environment variable, and no .env entry
    When configuration is loaded
    Then a clear error is reported and the process exits without attempting to load a model

  Scenario: A flag missing its value is rejected cleanly
    Given a flag such as "--max-tokens" with nothing following it
    When configuration is loaded
    Then a clear error naming that flag is reported, not an unhandled exception

  Scenario: A malformed numeric flag value is rejected cleanly
    Given "--max-tokens" followed by non-numeric text
    When configuration is loaded
    Then a clear error naming that flag and the bad value is reported, not an unhandled exception

  Scenario: A malformed numeric environment variable is rejected cleanly
    Given INFERENCE_MAX_TOKENS set to non-numeric text
    When configuration is loaded
    Then a clear error naming that variable and the bad value is reported

  Scenario: .env parsing tolerates comments and blank lines
    Given a .env file containing comment lines and blank lines alongside real entries
    When the file is loaded
    Then the real entries are applied and the comments and blank lines are ignored

  Scenario: .env parsing strips surrounding quotes from values
    Given a .env file with a quoted value
    When the file is loaded
    Then the value is applied without its surrounding quotes

  Scenario: .env parsing never overrides a variable already set in the real environment
    Given an environment variable already has a value
    And a .env file sets the same variable to a different value
    When the file is loaded
    Then the environment variable keeps its original value

  Scenario: A generation-time error is reported cleanly, not as a raw stack trace
    Given valid CLI configuration but an input that fails Engine-level validation
      (for example a negative --max-tokens)
    When the CLI runs
    Then the error message is printed and the process exits with a non-zero code,
      without an unhandled-exception stack trace

  Scenario: Output always reaches the real console streams
    Given the console's standard output and error streams are captured
    When the CLI writes normal output and an error message
    Then the normal output appears on standard output and the error message on standard error
