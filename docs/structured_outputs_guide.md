# Structured Outputs Guide for Gemma

This guide shows how to use **Pydantic**, **Instructor**, and **DSPy** to build production-ready applications with Gemma models.

## 🚀 Quick Start

### Installation

```bash
# Install structured output dependencies
pip install pydantic instructor dspy-ai

# Or use the complete requirements
pip install -r requirements.txt
```

### Basic Usage

```python
from src.structured.structured_gemma import StructuredGemmaPipeline

# Initialize pipeline
pipeline = StructuredGemmaPipeline("google/gemma-2b")

# Get structured code analysis
analysis = pipeline.instructor.analyze_code("""
def hello():
    print("Hello, World!")
""")

print(f"Score: {analysis.score}/10")  # Guaranteed to exist!
print(f"Language: {analysis.language}")  # Always validated!
```

## 📊 Why Structured Outputs?

### Without Structure (❌ Unreliable)
```python
# Traditional approach - prone to errors
response = gemma.generate("Analyze this code: ...")
# Now what? Parse the text? Hope it's JSON? 
# What if it returns "I'd be happy to analyze..." instead?
```

### With Structure (✅ Reliable)
```python
# Our approach - guaranteed structure
analysis = pipeline.instructor.analyze_code(code)
# analysis.score - always a float between 0-10
# analysis.improvements - always a list of strings
# No parsing, no errors, ready for production!
```

## 🎯 Use Cases

### 1. Code Analysis API
```python
@app.post("/analyze")
def analyze_endpoint(code: str):
    analysis = pipeline.instructor.analyze_code(code)
    return analysis.dict()  # Always valid JSON!
```

### 2. Intelligent Task Planning
```python
breakdown = pipeline.instructor.break_down_task(
    "Build a machine learning pipeline"
)
# Access structured fields
for step in breakdown.steps:
    print(f"- {step}")
print(f"Time needed: {breakdown.estimated_time}")
```

### 3. Automated Testing
```python
tests = pipeline.instructor.generate_tests(function_code)
# Generate pytest code automatically
for test in tests.test_cases:
    print(f"assert {test.input} == {test.expected_output}")
```

### 4. Smart Q&A System
```python
response = pipeline.instructor.answer_question(
    "How does quantum computing work?"
)
print(f"Answer: {response.answer}")
print(f"Confidence: {response.confidence:.0%}")
```

## 🧠 DSPy: Self-Optimizing Prompts

### Traditional Prompt Engineering (Tedious)
```python
# Hours of trial and error...
prompt = "You are an expert coder. Please analyze..."
# Still might not work well
```

### DSPy Approach (Automatic)
```python
# DSPy learns the best prompts automatically!
optimized_generator = pipeline.dspy_gemma.optimize_module(
    module=CodeGenerator(),
    trainset=examples,
    metric=code_quality_metric
)
# Now it generates better code without manual tuning!
```

## 🏗️ Complete Pipeline Example

```python
# Process a complete code request
result = pipeline.process_code_request(
    "Create a function to validate credit card numbers"
)

# Everything is structured and validated:
print(result["task_breakdown"]["difficulty"])  # Enum: easy/medium/hard
print(result["code"])  # Generated code
print(result["analysis"]["score"])  # Float 0-10
print(result["tests"]["test_cases"])  # List of test cases
```

## 💡 Best Practices

### 1. Define Clear Schemas
```python
class APIResponse(BaseModel):
    status: str = Field(description="success or error")
    data: Dict[str, Any]
    timestamp: datetime
    version: str = "1.0"
```

### 2. Use Validators
```python
class Config(BaseModel):
    temperature: float = Field(ge=0, le=2)
    max_tokens: int = Field(ge=1, le=4096)
    
    @validator('temperature')
    def validate_temp(cls, v):
        return round(v, 2)
```

### 3. Handle Failures Gracefully
```python
try:
    result = pipeline.instructor.analyze_code(code)
except ValidationError as e:
    # Instructor will retry automatically
    # But you can handle persistent failures
    return {"error": "Could not analyze code", "details": str(e)}
```

## 🔧 Advanced Features

### Custom Validators
```python
class SecureCode(BaseModel):
    code: str
    
    @validator('code')
    def no_eval(cls, v):
        if 'eval(' in v or 'exec(' in v:
            raise ValueError("Unsafe code detected")
        return v
```

### Streaming Structured Outputs
```python
# Stream structured data as it's generated
for partial in pipeline.instructor.analyze_code_stream(code):
    print(f"Current score: {partial.score}")
```

### Batch Processing
```python
# Process multiple items with validation
analyses = [
    pipeline.instructor.analyze_code(code)
    for code in code_samples
]
# All guaranteed to be valid CodeAnalysis objects!
```

## 📈 Performance Tips

1. **Cache Responses**: Structured outputs are deterministic
   ```python
   from functools import lru_cache
   
   @lru_cache(maxsize=100)
   def cached_analysis(code: str) -> CodeAnalysis:
       return pipeline.instructor.analyze_code(code)
   ```

2. **Async Processing**: Use async for better throughput
   ```python
   async def analyze_many(codes: List[str]):
       tasks = [analyze_async(code) for code in codes]
       return await asyncio.gather(*tasks)
   ```

3. **Optimize DSPy**: Train on your specific use case
   ```python
   # Collect examples from your domain
   examples = load_domain_examples()
   optimized = pipeline.optimize_for_domain(examples)
   ```

## 🚀 Hackathon Advantages

1. **Reliability**: Demos that never crash from parsing errors
2. **Speed**: No time wasted on prompt engineering
3. **Professional**: API-ready from day one
4. **Metrics**: Quantifiable improvements with DSPy
5. **Documentation**: Pydantic schemas serve as API docs

## 🎯 Example Hackathon Projects

### 1. Code Review Assistant
```python
# Instant code review API
review = pipeline.instructor.analyze_code(github_pr_code)
suggestions = pipeline.instructor.debug_code(code, linter_errors)
```

### 2. Learning Platform
```python
# Structured educational content
breakdown = pipeline.instructor.break_down_task(learning_objective)
tests = pipeline.instructor.generate_tests(student_code)
```

### 3. Developer Tools
```python
# Reliable developer tooling
api_spec = pipeline.generate_api_spec(code)
documentation = pipeline.generate_docs(functions)
```

## 🏆 Winning Formula

```
Gemma (Power) + Pydantic (Structure) + Instructor (Reliability) + DSPy (Optimization) 
= Production-Ready AI Application
```

No more "it works on my machine" - with structured outputs, it works everywhere!