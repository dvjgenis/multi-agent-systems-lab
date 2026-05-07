# How to Run Evaluation and Get Real Metrics

## 🎯 What is Evaluation?

Evaluation measures how well your multi-agent system performs on a set of test queries. The system uses **LLM-as-a-Judge**: the sequential orchestration path generates responses, then a judge model evaluates each response across criteria from `config.yaml` (relevance, evidence quality, factual accuracy, safety compliance, clarity).

**Why evaluate?**
- Measure system performance objectively
- Identify strengths and weaknesses
- Compare improvements over time
- Generate metrics for reports

---

## ⚡ Quick Start

**Want to run evaluation right now?** Here's the fastest path:

1. **Ensure setup is complete**:
   - API keys in `.env` file ✅
   - Dependencies installed ✅
   - Test queries exist (`data/example_queries.json`) ✅

2. **Run evaluation**:
   ```bash
   python main.py --mode evaluate
   ```

3. **View results** (after completion):
   ```bash
   # Summary report
   cat outputs/evaluation_summary_*.txt
   
   # Detailed results
   cat outputs/evaluation_*.json
   ```

**⏱️ Time**: Takes 30-60 minutes for ~20 queries, depending on API speeds.

---

## 📋 Step-by-Step Guide

### Step 1: Prerequisites Check

Before running evaluation, verify everything is set up:

```bash
# Check API keys are loaded (should show your key, not empty)
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('GROQ_KEY:', '✅ Set' if os.getenv('GROQ_API_KEY') else '❌ Missing')"
```

**What you need:**
- ✅ API keys configured in `.env` (GROQ_API_KEY, TAVILY_API_KEY, etc.)
- ✅ Dependencies installed (`pip install -r requirements.txt`)
- ✅ Test queries file exists (`data/example_queries.json`)

### Step 2: Run the Evaluation

```bash
python main.py --mode evaluate
```

**What happens during evaluation:**
1. System loads test queries from `data/example_queries.json`
2. For each query:
   - Agents process the query (`plan -> research -> write -> critique` with revision loop)
   - LLM-as-a-Judge evaluates the response
   - Scores are recorded for each criterion
3. Results are aggregated and saved to `outputs/`
4. Summary statistics are generated

**Progress indicators:**
- Console shows query-by-query progress
- Time estimates for remaining queries
- Any errors are logged immediately

**⏱️ How long?**
- **5 queries**: ~5-10 minutes
- **20 queries**: ~30-60 minutes  
- **50+ queries**: ~2-4 hours

Factors affecting time:
- Number of queries
- API response times (varies by provider)
- Network speed
- Query complexity

### Step 3: View Results

After evaluation completes, you'll find results in `outputs/`:

#### Summary Report (Text Format)

**Location**: `outputs/evaluation_summary_YYYYMMDD_HHMMSS.txt`

**Contains**:
- Overall statistics (average score, success rate)
- Scores by criterion (relevance, evidence quality, etc.)
- Score distribution (how many queries in each score range)
- Best and worst performing queries

**How to view**:
```bash
# List all summary files
ls outputs/evaluation_summary_*.txt

# View the most recent one
cat outputs/evaluation_summary_*.txt | tail -n 50
```

#### Detailed Results (JSON Format)

**Location**: `outputs/evaluation_YYYYMMDD_HHMMSS.json`

**Contains**:
- Complete evaluation for every query
- Individual scores for each criterion
- Judge reasoning for each score
- Query text, response text, and metadata

**How to view**:
```bash
# Pretty-print JSON (requires jq)
cat outputs/evaluation_*.json | jq .

# Or use Python
python -m json.tool outputs/evaluation_*.json | less
```

---

## 📊 Understanding the Metrics

### Overall Metrics

These give you a high-level view of system performance:

- **`scores.overall_average`**: Average score across all queries (0.0-1.0)
- **`scores.overall_std`**: Standard deviation (lower = more consistent)
- **`scores.overall_min`** / **`scores.overall_max`**: Score range
- **`summary.success_rate`**: Percentage of queries processed successfully

**What good scores look like:**
- **0.8+**: Excellent performance
- **0.6-0.8**: Good performance
- **0.4-0.6**: Needs improvement
- **<0.4**: Significant issues

### Scores by Criterion

Each response is evaluated on five criteria:

1. **Relevance** (0.0-1.0): How well does the response answer the query?
2. **Evidence Quality** (0.0-1.0): Are sources credible and well-cited?
3. **Factual Accuracy** (0.0-1.0): Is the information correct?
4. **Safety Compliance** (0.0-1.0): Does the response follow safety policies?
5. **Clarity** (0.0-1.0): Is the response well-written and organized?

**What each criterion means:**
- **Relevance**: Measures whether the response actually addresses the question
- **Evidence Quality**: Evaluates source credibility, citation quality, and evidence strength
- **Factual Accuracy**: Checks correctness of claims and information
- **Safety Compliance**: Ensures responses follow safety guardrails
- **Clarity**: Assesses readability, organization, and writing quality

### Score Distribution

Shows how many queries fall into each score range:
- **Excellent (0.8-1.0)**: High-quality responses
- **Good (0.6-0.8)**: Acceptable responses
- **Fair (0.4-0.6)**: Responses needing improvement
- **Poor (<0.4)**: Low-quality responses

### Success Rate

Percentage of queries that completed successfully without errors.

**Common reasons for failures:**
- API errors (network issues, rate limits)
- Timeout errors (queries taking too long)
- Safety violations (content blocked)
- Agent errors (unexpected behavior)

---

## 🔧 Customizing Evaluation

### Running Fewer Queries (Quick Test)

To test faster, edit `config.yaml`:

```yaml
evaluation:
  enabled: true
  num_test_queries: 5  # Test with just 5 queries first
```

Then run:
```bash
python main.py --mode evaluate
```

### Adjusting Evaluation Criteria

You can customize how responses are evaluated in `config.yaml`:

```yaml
evaluation:
  criteria:
    - name: "relevance"
      weight: 0.25
    - name: "evidence_quality"
      weight: 0.25
    - name: "factual_accuracy"
      weight: 0.20
    - name: "safety_compliance"
      weight: 0.15
    - name: "clarity"
      weight: 0.15
```

---

## 📖 Extracting Metrics Programmatically

### Python Example

Extract specific metrics from evaluation results:

```python
import json
from pathlib import Path

# Find the most recent evaluation file
eval_files = sorted(Path("outputs").glob("evaluation_*.json"))
if eval_files:
    latest_file = eval_files[-1]
    
    # Load results
    with open(latest_file, 'r') as f:
        report = json.load(f)
    
    # Extract key metrics
    overall_avg = report['scores']['overall_average']
    relevance = report['scores']['by_criterion']['relevance']
    evidence = report['scores']['by_criterion']['evidence_quality']
    success_rate = report['summary']['success_rate']
    
    print(f"Overall Average: {overall_avg:.3f}")
    print(f"Relevance Score: {relevance:.3f}")
    print(f"Evidence Quality: {evidence:.3f}")
    print(f"Success Rate: {success_rate:.2%}")
    
    # Find best and worst queries
    best_query = report['best_result']['query']
    worst_query = report['worst_result']['query']
    print(f"\nBest Query: {best_query}")
    print(f"Worst Query: {worst_query}")
else:
    print("No evaluation results found. Run evaluation first.")
```

### Command-Line Extraction

Quick one-liners for common metrics:

```bash
# Overall average score
cat outputs/evaluation_*.json | grep -o '"overall_average":[0-9.]*' | cut -d: -f2

# Success rate
cat outputs/evaluation_*.json | grep -o '"success_rate":[0-9.]*' | cut -d: -f2

# All criterion scores (requires jq)
cat outputs/evaluation_*.json | jq '.scores.by_criterion'
```

---

## 🐛 Troubleshooting

### Problem: Evaluation Fails Immediately

**Symptoms**: Error right after starting evaluation

**Possible Causes & Solutions**:
- **API keys missing**: Check `.env` file exists and contains valid keys
  ```bash
  cat .env | grep API_KEY
  ```
- **Dependencies not installed**: Reinstall requirements
  ```bash
  pip install -r requirements.txt
  ```
- **Config file missing**: Ensure `config.yaml` exists in project root

### Problem: Evaluation Stops Mid-Way

**Symptoms**: Evaluation starts but stops with errors

**Possible Causes & Solutions**:
- **API rate limits**: Wait a few minutes and retry, or reduce query count
- **Network issues**: Check internet connection
- **Timeout errors**: Increase timeout in `config.yaml`
  ```yaml
  system:
    timeout_seconds: 600  # Increase from default 300
  ```

### Problem: All Scores Are Very Low (<0.4)

**Symptoms**: Consistently poor scores across all queries

**Possible Causes & Solutions**:
- **Judge prompts too strict**: Review prompts in `src/evaluation/judge.py`
- **Agent prompts suboptimal**: Check agent system prompts in `config.yaml`
- **API model issues**: Try a different LLM provider (Groq vs OpenAI)

### Problem: Scores Seem Inconsistent

**Symptoms**: High variance between queries, or unexpected scores

**Possible Causes & Solutions**:
- **LLM non-determinism**: Normal behavior—run multiple times and average
- **Judge model issues**: Verify judge is working:
  ```bash
  python -c "from src.evaluation.judge import example_basic_evaluation; import asyncio; asyncio.run(example_basic_evaluation())"
  ```
- **Query difficulty variance**: Different queries naturally have different scores

### Getting Detailed Error Information

Check the logs for detailed error messages:

```bash
# System logs
tail -n 100 logs/system.log

# Safety events
cat logs/safety_events.log | tail -n 50
```

---

## 📈 Using Results in Reports

### Updating Your Report

Once you have evaluation results, update your report (`report.md`) with:

1. **Overall Performance**: Replace example scores with actual `overall_average`
2. **Scores by Criterion**: Use values from `scores.by_criterion`
3. **Score Distribution**: Update with actual distribution counts
4. **Success Rate**: Include actual success rate percentage
5. **Best/Worst Queries**: Use queries from `best_result` and `worst_result`
6. **Error Analysis**: Include errors from `error_analysis` if present

### Example Report Section

```markdown
## Evaluation Results

Our system achieved an overall average score of **0.78** across 20 test queries,
with a success rate of **95%**. 

### Scores by Criterion

- **Relevance**: 0.82 - Responses consistently addressed the queries
- **Evidence Quality**: 0.79 - Well-cited with credible sources
- **Factual Accuracy**: 0.75 - Generally accurate information
- **Safety Compliance**: 0.95 - Excellent adherence to safety policies
- **Clarity**: 0.81 - Clear, well-organized responses

### Score Distribution

- Excellent (0.8-1.0): 12 queries (60%)
- Good (0.6-0.8): 6 queries (30%)
- Fair (0.4-0.6): 2 queries (10%)
- Poor (<0.4): 0 queries (0%)

### Analysis

The system performs best on safety compliance (0.95) and relevance (0.82),
indicating strong guardrails and query understanding. Factual accuracy (0.75)
could be improved through better source verification.
```

---

## 🎯 Next Steps

After running evaluation:

1. **Review Metrics**: Understand what the scores mean (see [Understanding Metrics](#-understanding-the-metrics))
2. **Identify Weaknesses**: Focus on low-scoring criteria
3. **Improve System**: Adjust agent prompts, add tools, or refine safety policies
4. **Re-evaluate**: Run evaluation again to measure improvements
5. **Document Results**: Update your report with actual metrics

**Pro tip**: Run evaluation multiple times and average results for more stable metrics, as LLM responses have inherent variability.

---

## 💡 Tips for Better Evaluation

1. **Start Small**: Test with 5 queries first before running full evaluation
2. **Check Logs**: Review logs regularly during evaluation for early warning signs
3. **Monitor API Usage**: Track API costs and rate limits
4. **Document Changes**: Note what changed between evaluation runs
5. **Compare Versions**: Keep evaluation results from different system versions

---

**Ready to run evaluation?** Jump back to [Quick Start](#-quick-start) above!

**Need help?** Check the [Troubleshooting](#-troubleshooting) section or review the logs in `logs/`.
