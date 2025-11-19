using LLM_Module_API.Services;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;

namespace LLM_Module_API.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class MyEmbeddingController : ControllerBase
    {
        private readonly IEmbeddingService _embeddingService;

        public MyEmbeddingController(IEmbeddingService embeddingService)
        {
            this._embeddingService = embeddingService;
        }

        [HttpPost("embed")]
        public IActionResult GetEmbedding([FromBody] List<int> tokenIds)
        {

            var embeddings = _embeddingService.GetEmbeddings(tokenIds);
            return Ok( new { Tokens = tokenIds, Embedding = embeddings });
        }

    }
}
