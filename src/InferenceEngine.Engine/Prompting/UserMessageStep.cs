namespace InferenceEngine.Engine.Prompting;

internal sealed class UserMessageStep(string userPrompt) : IChatPromptStep
{
    public void Apply(ChatPromptBuilder builder)
    {
        builder.AddSpecial("<|im_start|>");
        builder.AddText("user\n");
        builder.AddText(userPrompt);
        builder.AddSpecial("<|im_end|>");
        builder.AddText("\n");
    }
}
