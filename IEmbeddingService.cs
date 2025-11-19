namespace LLM_Module_API.Services
{
    public interface IEmbeddingService
    {
        float[][] GetEmbeddings(List<int> tokenIds);
    }
}
