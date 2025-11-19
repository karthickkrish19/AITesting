using LLM_Module_API.Models;
using LLM_Module_API.Services;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace LLM_Module_API.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class MyTokeniserController : ControllerBase
    {

        private readonly ITokeniserService _tokeniserService;

        public MyTokeniserController(ITokeniserService tokeniserService)
        {
            _tokeniserService = tokeniserService;
        }

        [HttpGet("train")]
        public IActionResult Train()
        {
            var vocab = _tokeniserService.TrainTokeniser();

            return Ok(new { Vocab = vocab, Message = vocab.Count > 0 ? "Saved successfully" : "Save Failed" });
        }

        [HttpGet("load")]
        [ProducesResponseType(typeof(TokeniserLoadResponse), StatusCodes.Status200OK)]
        public IActionResult Load() {
            var (vocab, idToToken, merges) = _tokeniserService.Load();
            var mergeList = merges.Select(m => new MergePair { First = m.Item1, Second = m.Item2 }).ToList();
            return Ok(new TokeniserLoadResponse
            {
                Vocab = vocab,
                IdToToken = idToToken,
                Merges = mergeList
            });
        }

        [HttpPost("encode")]
        public IActionResult Encode([FromBody] string text) 
        {
            var tokenid = _tokeniserService.Encode(text).Where(x => x != 4).ToList();
            
            return Ok(tokenid);
        }

        [HttpPost("decode")]
        public IActionResult Decode([FromBody] List<int> tokenIds) => Ok(_tokeniserService.Decode(tokenIds));

        [HttpGet("vocabsize")]
        public IActionResult VocabSize() => Ok(_tokeniserService.GetVocabSize());

    }
}
