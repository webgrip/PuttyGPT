using Grpc.Core;
using Standard.AI.OpenAI.Clients.OpenAIs;
using Standard.AI.OpenAI.Models.Configurations;
using Standard.AI.OpenAI.Models.Services.Foundations.ChatCompletions;
using WebGrip.Protos;

namespace WebGrip.Putty.Brazen.Services
{
    public class ChatService : Protos.ChatService.ChatServiceBase
    {
        private readonly ILogger<ChatService> _logger;

        public ChatService(ILogger<ChatService> logger)
        {
            _logger = logger;
        }

        public override async Task<QuestionResponse> AskQuestion(QuestionRequest request, ServerCallContext context)
        {
            _logger.LogDebug(request.ToString());


            var response = new QuestionResponse();

            try
            {
                // do request

                response.Status = "1";
                response.Message = "result";
            }
            catch (Exception ex)
            {
                response.Status = "2";

                _logger.LogError(ex.ToString());
                _logger.LogError($"Error doing request for question: {ex.Message}");

                var openAIConfigurations = new OpenAIConfigurations
                {
                    ApiKey = "YOUR_API_KEY_HERE", // add config
                    OrganizationId = "YOUR_OPTIONAL_ORG_ID_HERE" //optional
                };

                var openAIClient = new OpenAIClient(openAIConfigurations);

                var chatCompletion = new ChatCompletion
                {
                    Request = new ChatCompletionRequest
                    {
                        Model = "gpt-3.5-turbo",
                        Messages = new ChatCompletionMessage[]
                        {
                        new ChatCompletionMessage
                        {
                            
                            Content = "What is c#?",
                            Role = "user",
                        }
                        },
                    }
                };

                ChatCompletion resultChatCompletion = await openAIClient.ChatCompletions.SendChatCompletionAsync(chatCompletion);

                Array.ForEach(
                    resultChatCompletion.Response.Choices,
                    choice => 
                        Console.WriteLine(value: $"{choice.Message.Role}: {choice.Message.Content}")
                );

                //var errorMessages = new Dictionary<string, string>
                //{
                //    { GraphErrorCode.InvalidRequest.ToString(), "Invalid request. Please check the provided user data." },
                //    { GraphErrorCode.AuthenticationFailure.ToString(), "Authentication failed. Check the credentials and required scopes." },
                //    { GraphErrorCode.GeneralException.ToString(), "A network error or service outage occurred. Please try again later." },
                //    { GraphErrorCode.ServiceNotAvailable.ToString(), "A network error or service outage occurred. Please try again later." }
                //};

                //response.Message = errorMessages.TryGetValue(ex.ResponseStatusCode.ToString(), out var message) ? message : $"An unknown error occurred: {ex.Message}";

                response.Message = $"Unexpected error: {ex.Message}";
            }

            return response;
        }
    }
}