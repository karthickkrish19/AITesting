using System.Collections.Concurrent;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace LLM_Module_API.Services
{
    public class TokeniserService : ITokeniserService
    {
        private readonly string _inputDir;
        private readonly string _outputDir;
        private readonly string _textFilePath;
        private int vocabSize = 50000;
        private readonly List<string> specialTokens = new() { "<unk>", "<pad>", "<bos>", "<eos>", "</w>" };
        private Dictionary<string, int> vocab = new();
        private Dictionary<int, string> idToToken = new();
        private List<(string, string)> merges = new();
        private Dictionary<(string, string), int> ranks = new();
        private bool isLoaded = false;

        public TokeniserService(IWebHostEnvironment env)
        {
            _inputDir = Path.Combine(env.WebRootPath, "data", "input");
            _outputDir = Path.Combine(env.WebRootPath, "data", "output");
            _textFilePath = Path.Combine(_inputDir, "sampleinput.txt");

            if (File.Exists(Path.Combine(_outputDir, "vocab.json")))
                Load();

        }

        public Dictionary<string, int> TrainTokeniser()
        {
            if (!File.Exists(_textFilePath))
                throw new FileNotFoundException("Input file not found", _textFilePath);

            string content = File.ReadAllText(_textFilePath);
            string cleanedText = ClearText(content);

            var tokenLists = cleanedText
                .Split(' ', StringSplitOptions.RemoveEmptyEntries)
                .Select(word => word.Select(c => c.ToString()).ToArray())
                .ToList();

            var corpus = tokenLists;
            var tokenSet = new HashSet<string>();

            while (tokenSet.Count < vocabSize)
            {
                var pairFreqs = GetPairFrequenciesParallel(corpus);
                if (pairFreqs.Count == 0) break;

                // Find best pair without full sort
                var bestPair = pairFreqs.Aggregate((l, r) => l.Value > r.Value ? l : r).Key;

                merges.Add(bestPair);
                corpus = MergePair(bestPair, corpus);

                foreach (var word in corpus)
                    foreach (var t in word)
                        tokenSet.Add(t);

                if (tokenSet.Count >= vocabSize) break;
            }

            var fullVocab = specialTokens.Concat(tokenSet.OrderBy(t => t)).ToList();
            vocab = fullVocab.Select((tok, i) => new { tok, i }).ToDictionary(x => x.tok, x => x.i);
            idToToken = vocab.ToDictionary(kv => kv.Value, kv => kv.Key);
            ranks = merges.Select((m, i) => new { m, i }).ToDictionary(x => x.m, x => x.i);

            if (vocab.Count > 0)
                SaveTokenAsync().Wait();

            isLoaded = true;

            return vocab;
        }

        private async Task SaveTokenAsync()
        {
            if (!Directory.Exists(_outputDir))
                Directory.CreateDirectory(_outputDir);

            await File.WriteAllTextAsync(Path.Combine(_outputDir, "vocab.json"), JsonSerializer.Serialize(vocab));
            var mergeLines = new[] { "#v0.1" }.Concat(merges.Select(m => $"{m.Item1} {m.Item2}"));
            await File.WriteAllLinesAsync(Path.Combine(_outputDir, "merges.txt"), mergeLines);
        }

        public (Dictionary<string, int> Vocab, Dictionary<int, string> IdToToken, List<(string, string)> Merges) Load()
        {

            if (!isLoaded)
            {
                vocab = JsonSerializer.Deserialize<Dictionary<string, int>>(File.ReadAllText(Path.Combine(_outputDir, "vocab.json")));
                idToToken = vocab.ToDictionary(kv => kv.Value, kv => kv.Key);
                var lines = File.ReadAllLines(Path.Combine(_outputDir, "merges.txt")).Skip(1);
                merges = lines.Select(l => { var p = l.Split(' '); return (p[0], p[1]); }).ToList();
                ranks = merges.Select((m, i) => new { m, i }).ToDictionary(x => x.m, x => x.i);
                isLoaded = true;
            }
            return (vocab, idToToken, merges);
        }

        public List<int> Encode(string text)
        {
            string cleanedText = ClearText(text);
            if (!isLoaded) Load();

            var tokenIds = new List<int>();
            var unknowns = new List<string>();

            foreach (var word in cleanedText.Split(' '))
            {
                var tokens = word.Select(c => c.ToString()).ToList();
                tokens.Add("</w>");
                tokens = ApplyMerges(tokens);

                foreach (var t in tokens)
                {
                    if (vocab.ContainsKey(t))
                        tokenIds.Add(vocab[t]);
                    else
                    {
                        tokenIds.Add(vocab["<unk>"]);
                        unknowns.Add(t);
                    }
                }
            }
            return tokenIds;
        }

        public string Decode(List<int> tokenIds)
        {
            if (!isLoaded) Load();
            var tokens = tokenIds.Select(id => idToToken.ContainsKey(id) ? idToToken[id] : "<unk>").ToList();
            string word = "", result = "";
            foreach (var t in tokens)
            {
                if (t.EndsWith("</w>"))
                {
                    result += word + t.Replace("</w>", " ");
                    word = "";
                }
                else word += t;
            }
            return result.Trim();
        }

        public int GetVocabSize() 
        {
            Load();
            return vocab.Count();
         }

        // Optimized Pair Frequency Calculation
        private ConcurrentDictionary<(string, string), int> GetPairFrequenciesParallel(List<string[]> corpus)
        {
            var pairs = new ConcurrentDictionary<(string, string), int>();
            Parallel.ForEach(corpus, word =>
            {
                for (int i = 0; i < word.Length - 1; i++)
                {
                    var pair = (word[i], word[i + 1]);
                    pairs.AddOrUpdate(pair, 1, (_, old) => old + 1);
                }
            });
            return pairs;
        }

        private List<string[]> MergePair((string, string) pair, List<string[]> corpus)
        {
            var result = new List<string[]>();
            foreach (var word in corpus)
            {
                var newWord = new List<string>();
                int i = 0;
                while (i < word.Length)
                {
                    if (i < word.Length - 1 && (word[i], word[i + 1]) == pair)
                    {
                        newWord.Add(word[i] + word[i + 1]);
                        i += 2;
                    }
                    else newWord.Add(word[i++]);
                }
                result.Add(newWord.ToArray());
            }
            return result;
        }

        private List<string> ApplyMerges(List<string> tokens)
        {
            while (true)
            {
                var pairs = Enumerable.Range(0, tokens.Count - 1).Select(i => (tokens[i], tokens[i + 1])).ToList();
                if (!pairs.Any()) break;
                var ranked = pairs.Select(p => (p, ranks.ContainsKey(p) ? ranks[p] : int.MaxValue)).ToList();
                if (ranked.All(x => x.Item2 == int.MaxValue)) break;
                var bestPair = ranked.OrderBy(x => x.Item2).First().p;
                var newTokens = new List<string>();
                int i = 0;
                while (i < tokens.Count)
                {
                    if (i < tokens.Count - 1 && (tokens[i], tokens[i + 1]) == bestPair)
                    {
                        newTokens.Add(tokens[i] + tokens[i + 1]);
                        i += 2;
                    }
                    else newTokens.Add(tokens[i++]);
                }
                tokens = newTokens;
            }
            return tokens;
        }

        private string ClearText(string text)
        {
            if (string.IsNullOrEmpty(text)) return string.Empty;
            text = Regex.Replace(text, "<.*?>", string.Empty);
            text = Regex.Replace(text, @"https?://\S+\nwww\.\S+", string.Empty);
            text = System.Text.Encoding.ASCII.GetString(System.Text.Encoding.ASCII.GetBytes(text));
            text = Regex.Replace(text, @"[^\w\s.,?!]", string.Empty);
            text = Regex.Replace(text, @"\d+", string.Empty);
            text = Regex.Replace(text, @"\s+", " ").Trim().ToLower();
            return text;
        }
    }
}
