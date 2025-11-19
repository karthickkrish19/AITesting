namespace LLM_Module_API.Services
{
    public class EmbeddingService : IEmbeddingService
    {
        private readonly int vocabSize;
        private readonly int embeddingDim;
        private readonly int maxSeqLength;
        private readonly float[,] tokenEmbeddingMatrix;
        private readonly float[,] positionEmbeddingMatrix;
        private readonly Random rand = new Random();

        public EmbeddingService(int vocabSize, int embeddingDim, int maxSeqLength)
        {

            this.vocabSize = vocabSize;
            this.embeddingDim = embeddingDim;
            this.maxSeqLength = maxSeqLength;

            tokenEmbeddingMatrix = new float[vocabSize, embeddingDim];
            positionEmbeddingMatrix = new float[maxSeqLength, embeddingDim];

            InitializeEmbeddings(tokenEmbeddingMatrix);
            InitializeEmbeddings(positionEmbeddingMatrix);

        }


        // GPT-style initialization: Normal(0, std) with std = 1/sqrt(embeddingDim)
        private void InitializeEmbeddings(float[,] matrix)
        {
            double std = 1.0 / Math.Sqrt(embeddingDim);
            for (int i = 0; i < matrix.GetLength(0); i++)
            {
                for (int j = 0; j < matrix.GetLength(1); j++)
                {
                    matrix[i, j] = (float)(SampleNormal() * std);
                }
            }
        }


        // Box-Muller transform for normal distribution
        private double SampleNormal()
        {
            double u1 = 1.0 - rand.NextDouble();
            double u2 = 1.0 - rand.NextDouble();
            return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        }

        public float[][] GetEmbeddings(List<int> tokenIds)
        {
            if (tokenIds.Count > maxSeqLength)
                throw new ArgumentException($"Sequence length exceeds max {maxSeqLength}");

            var embeddings = new float[tokenIds.Count][];
            for (int i = 0; i < tokenIds.Count; i++)
            {
                embeddings[i] = new float[embeddingDim];
                for (int j = 0; j < embeddingDim; j++)
                {
                    // Token embedding + positional embedding
                    embeddings[i][j] = tokenEmbeddingMatrix[tokenIds[i], j] + positionEmbeddingMatrix[i, j];
                }
            }

            return embeddings;
        }
    }
}
