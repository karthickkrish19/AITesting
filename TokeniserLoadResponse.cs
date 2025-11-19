namespace LLM_Module_API.Models
{
    public class TokeniserResult
    {
    }


    public class TokeniserLoadResponse
    {
        public Dictionary<string, int> Vocab { get; set; }
        public Dictionary<int, string> IdToToken { get; set; }
        public List<MergePair> Merges { get; set; }
    }

    public class MergePair
    {
        public string First { get; set; }
        public string Second { get; set; }
    }


}
