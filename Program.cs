
using LLM_Module_API.Services;

namespace LLM_Module_API
{
    public class Program
    {
        public static void Main(string[] args)
        {
            var builder = WebApplication.CreateBuilder(args);

            // Add services to the container.
            builder.Services.AddSingleton<ITokeniserService, TokeniserService>();
            builder.Services.AddSingleton<IEmbeddingService>(sp =>
            {
                var tokeniser = sp.GetRequiredService<ITokeniserService>();
                int vocabSize = tokeniser.GetVocabSize();
                int embeddingDim = 3;
                int maxSeqLength = 512;
                return new EmbeddingService(vocabSize, embeddingDim, maxSeqLength);
            });
            builder.Services.AddControllers();
            // Learn more about configuring Swagger/OpenAPI at https://aka.ms/aspnetcore/swashbuckle
            builder.Services.AddEndpointsApiExplorer();
            builder.Services.AddSwaggerGen();

            var app = builder.Build();

            // Configure the HTTP request pipeline.
            if (app.Environment.IsDevelopment())
            {
                app.UseSwagger();
                app.UseSwaggerUI();
            }

            app.UseStaticFiles();

            app.UseHttpsRedirection();

            app.UseAuthorization();


            app.MapControllers();

            app.Run();
        }
    }
}
