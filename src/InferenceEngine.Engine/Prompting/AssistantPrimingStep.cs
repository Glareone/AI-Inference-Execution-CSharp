namespace InferenceEngine.Engine.Prompting;

/// <summary>Opens the assistant's turn, priming the model to generate its response next.</summary>
internal sealed class AssistantPrimingStep : IChatPromptStep
{
    public void Apply(ChatPromptBuilder builder)
    {
        builder.AddSpecial("<|im_start|>");
        builder.AddText("assistant\n");
    }
}
