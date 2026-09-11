namespace InferenceEngine.Engine.Prompting;

/// <summary>One link in the chat-template chain — a single turn's contribution to the token sequence handed to the model.</summary>
internal interface IChatPromptStep
{
    void Apply(ChatPromptBuilder builder);
}
