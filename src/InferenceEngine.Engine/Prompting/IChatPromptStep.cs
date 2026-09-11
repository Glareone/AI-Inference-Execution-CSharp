namespace InferenceEngine.Engine.Prompting;

internal interface IChatPromptStep
{
    void Apply(ChatPromptBuilder builder);
}
