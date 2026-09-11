namespace InferenceEngine.Engine.Prompting;

internal sealed class SystemMessageStep(string systemPrompt) : IChatPromptStep
{
    public void Apply(ChatPromptBuilder builder)
    {
        builder.AddSpecial("<|im_start|>");
        builder.AddText("system\n");
        builder.AddText(systemPrompt);
        builder.AddSpecial("<|im_end|>");
        builder.AddText("\n");
    }
}
