using LLama;
using LLama.Native;
using LLama.Common;
using LLamaSharp.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.Extensions.DependencyInjection;
using ChatHistory = Microsoft.SemanticKernel.ChatCompletion.ChatHistory;

// Config
var modelPath = "HFModel/mistral-7b-instruct-v0.2.Q5_K_S.gguf";

// Disable verbose logging
NativeLogConfig.llama_log_set((level, message) =>
{
  if (level == LLamaLogLevel.Error || level == LLamaLogLevel.Warning)
  {
    System.Console.WriteLine($"[{level}] {message}");
  }
});

// Prepare LLamaSharp
var parameters = new ModelParams(modelPath)
{
  ContextSize = 1024,
  GpuLayerCount = 5,
};
using var model = LLamaWeights.LoadFromFile(parameters);
var executor = new StatelessExecutor(model, parameters);

// Prepare SemanticKernel
Func<IServiceProvider, object?, IChatCompletionService> factory = (serviceProvider, _) => new LLamaSharpChatCompletion(executor);

IKernelBuilder builder = Kernel.CreateBuilder();
builder.Services.AddKeyedSingleton("local", factory);
Kernel kernel = builder.Build();

IChatCompletionService chatCompletionService = kernel.GetRequiredService<IChatCompletionService>();

// Create the chat history
ChatHistory chatMessages = new ChatHistory("""
You are a friendly assistant that can have chit chats.
""");

// Start the conversation
while (true)
{
  // Get user input
  System.Console.Write("User > ");
  var message = Console.ReadLine();
  if (message == "exit")
  {
    break;
  }

  chatMessages.AddUserMessage(message!);

  var result = chatCompletionService.GetStreamingChatMessageContentsAsync(
      chatMessages,
      kernel: kernel);

  string fullMessage = "";
  System.Console.Write("Assistant > ");
  await foreach (var content in result)
  {
    System.Console.Write(content.Content);
    fullMessage += content.Content;
  }
  System.Console.WriteLine();

  chatMessages.AddAssistantMessage(fullMessage);
}
