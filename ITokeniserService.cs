namespace LLM_Module_API.Services
{
    public interface ITokeniserService
    {
        Dictionary<string, int> TrainTokeniser();
        List<int> Encode(string text);
        int GetVocabSize();
        (Dictionary<string, int> Vocab, Dictionary<int, string> IdToToken, List<(string, string)> Merges) Load();
        string Decode(List<int> tokenIds);

    }
}
