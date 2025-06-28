"""
Structured Output Generation with Gemma using Pydantic, Instructor, and DSPy

This module provides reliable structured outputs and self-optimizing prompts
for production-ready applications.
"""

import json
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from datetime import datetime

from pydantic import BaseModel, Field, validator
import instructor
import dspy
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from ..models.gemma_model import GemmaModel


# ============== Pydantic Models for Structured Outputs ==============

class Difficulty(str, Enum):
    """Difficulty levels for tasks."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


class CodeAnalysis(BaseModel):
    """Structured output for code analysis."""
    language: str = Field(description="Programming language")
    purpose: str = Field(description="What the code does")
    complexity: str = Field(description="Time complexity (e.g., O(n))")
    space_complexity: str = Field(description="Space complexity")
    strengths: List[str] = Field(description="Code strengths")
    improvements: List[str] = Field(description="Suggested improvements")
    security_issues: Optional[List[str]] = Field(default=None, description="Security concerns")
    score: float = Field(ge=0, le=10, description="Overall quality score")
    
    @validator('score')
    def validate_score(cls, v):
        return round(v, 1)


class TaskBreakdown(BaseModel):
    """Structured task analysis and planning."""
    task_type: str = Field(description="Type of task (coding, analysis, creative, etc.)")
    difficulty: Difficulty
    estimated_time: str = Field(description="Estimated completion time")
    prerequisites: List[str] = Field(description="Required knowledge/tools")
    steps: List[str] = Field(description="Step-by-step breakdown")
    potential_challenges: List[str] = Field(description="Potential difficulties")
    success_criteria: List[str] = Field(description="How to measure success")


class QAResponse(BaseModel):
    """Structured Q&A response with citations."""
    answer: str = Field(description="Direct answer to the question")
    confidence: float = Field(ge=0, le=1, description="Confidence score")
    reasoning: str = Field(description="Step-by-step reasoning")
    sources: Optional[List[str]] = Field(default=None, description="Information sources")
    related_topics: List[str] = Field(description="Related topics to explore")
    
    @validator('confidence')
    def round_confidence(cls, v):
        return round(v, 2)


class DebugAnalysis(BaseModel):
    """Structured debugging output."""
    error_type: str = Field(description="Type of error")
    error_location: Optional[str] = Field(description="Where the error occurs")
    root_cause: str = Field(description="Root cause analysis")
    fix_suggestion: str = Field(description="How to fix the issue")
    code_fix: Optional[str] = Field(description="Corrected code")
    prevention_tips: List[str] = Field(description="How to prevent similar issues")


class TestCase(BaseModel):
    """Individual test case."""
    input: str
    expected_output: str
    edge_case: bool = False
    description: Optional[str] = None


class TestSuite(BaseModel):
    """Structured test generation output."""
    function_name: str
    test_cases: List[TestCase]
    edge_cases_covered: List[str]
    test_strategy: str
    coverage_estimate: float = Field(ge=0, le=100)


# ============== Instructor-wrapped Gemma Model ==============

class InstructorGemma:
    """
    Gemma model wrapped with Instructor for guaranteed structured outputs.
    """
    
    def __init__(self, model: GemmaModel):
        """
        Initialize Instructor-wrapped Gemma.
        
        Args:
            model: Base GemmaModel instance
        """
        self.model = model
        self.client = instructor.patch(
            create=self._create_completion,
            mode=instructor.Mode.JSON_SCHEMA,
        )
        
    def _create_completion(self, messages: List[Dict], **kwargs):
        """
        Adapter to make Gemma compatible with Instructor.
        """
        # Extract the prompt from messages
        prompt = messages[-1]["content"] if messages else ""
        
        # Generate with Gemma
        response = self.model.generate(prompt, max_new_tokens=512)[0]
        
        # Return in OpenAI-compatible format
        return {
            "choices": [{
                "message": {
                    "content": response
                }
            }]
        }
    
    def analyze_code(self, code: str) -> CodeAnalysis:
        """
        Analyze code with structured output.
        
        Args:
            code: Source code to analyze
            
        Returns:
            Structured CodeAnalysis object
        """
        return self.client.create(
            model="gemma",
            messages=[{
                "role": "user",
                "content": f"Analyze this code:\n```\n{code}\n```"
            }],
            response_model=CodeAnalysis,
        )
    
    def break_down_task(self, task: str) -> TaskBreakdown:
        """
        Break down a task into structured steps.
        
        Args:
            task: Task description
            
        Returns:
            Structured TaskBreakdown object
        """
        return self.client.create(
            model="gemma",
            messages=[{
                "role": "user",
                "content": f"Break down this task into detailed steps: {task}"
            }],
            response_model=TaskBreakdown,
        )
    
    def answer_question(self, question: str, context: Optional[str] = None) -> QAResponse:
        """
        Answer a question with structured response.
        
        Args:
            question: The question to answer
            context: Optional context information
            
        Returns:
            Structured QAResponse object
        """
        prompt = f"Question: {question}"
        if context:
            prompt = f"Context: {context}\n\n{prompt}"
            
        return self.client.create(
            model="gemma",
            messages=[{"role": "user", "content": prompt}],
            response_model=QAResponse,
        )
    
    def debug_code(self, code: str, error_message: str) -> DebugAnalysis:
        """
        Debug code with structured analysis.
        
        Args:
            code: Code with error
            error_message: Error message received
            
        Returns:
            Structured DebugAnalysis object
        """
        return self.client.create(
            model="gemma",
            messages=[{
                "role": "user",
                "content": f"Debug this code:\n```\n{code}\n```\nError: {error_message}"
            }],
            response_model=DebugAnalysis,
        )
    
    def generate_tests(self, function_code: str) -> TestSuite:
        """
        Generate test cases for a function.
        
        Args:
            function_code: Function to test
            
        Returns:
            Structured TestSuite object
        """
        return self.client.create(
            model="gemma",
            messages=[{
                "role": "user",
                "content": f"Generate comprehensive test cases for:\n```\n{function_code}\n```"
            }],
            response_model=TestSuite,
        )


# ============== DSPy Integration for Self-Optimizing Prompts ==============

class GemmaLM(dspy.LM):
    """
    DSPy-compatible Gemma language model.
    """
    
    def __init__(self, model: GemmaModel):
        self.model = model
        self.history = []
        
    def __call__(self, prompt: str, **kwargs) -> str:
        """Generate text with Gemma."""
        response = self.model.generate(prompt, **kwargs)[0]
        self.history.append((prompt, response))
        return response
    
    def get_convo(self) -> str:
        """Get conversation history."""
        return "\n".join([f"User: {p}\nAssistant: {r}" for p, r in self.history])


# DSPy Modules for Complex Tasks

class CodeGenerator(dspy.Module):
    """
    DSPy module for optimized code generation.
    """
    
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("requirements -> code")
        
    def forward(self, requirements: str) -> dspy.Prediction:
        return self.prog(requirements=requirements)


class CodeReviewer(dspy.Module):
    """
    DSPy module for code review with reasoning.
    """
    
    def __init__(self):
        super().__init__()
        self.review = dspy.ChainOfThought("code -> review")
        self.suggest = dspy.ChainOfThought("code, issues -> improvements")
        
    def forward(self, code: str) -> dspy.Prediction:
        review = self.review(code=code)
        improvements = self.suggest(code=code, issues=review.review)
        return dspy.Prediction(
            review=review.review,
            improvements=improvements.improvements
        )


class MultiStepReasoner(dspy.Module):
    """
    DSPy module for complex multi-step reasoning.
    """
    
    def __init__(self, num_steps: int = 3):
        super().__init__()
        self.steps = []
        for i in range(num_steps):
            self.steps.append(dspy.ChainOfThought(f"step_{i}_input -> step_{i}_output"))
            
    def forward(self, problem: str) -> dspy.Prediction:
        current = problem
        reasoning_chain = []
        
        for i, step in enumerate(self.steps):
            result = step(**{f"step_{i}_input": current})
            output = result[f"step_{i}_output"]
            reasoning_chain.append(output)
            current = output
            
        return dspy.Prediction(
            final_answer=current,
            reasoning_steps=reasoning_chain
        )


class StructuredDSPyGemma:
    """
    Combine DSPy optimization with structured outputs.
    """
    
    def __init__(self, model: GemmaModel):
        self.model = model
        self.lm = GemmaLM(model)
        dspy.settings.configure(lm=self.lm)
        
        # Initialize DSPy modules
        self.code_generator = CodeGenerator()
        self.code_reviewer = CodeReviewer()
        self.reasoner = MultiStepReasoner()
        
    def optimize_module(self, module: dspy.Module, trainset: List[dspy.Example],
                       metric: callable, trials: int = 10) -> dspy.Module:
        """
        Optimize a DSPy module with training data.
        
        Args:
            module: DSPy module to optimize
            trainset: Training examples
            metric: Evaluation metric
            trials: Number of optimization trials
            
        Returns:
            Optimized module
        """
        # Use BootstrapFewShot optimizer
        optimizer = dspy.BootstrapFewShot(
            metric=metric,
            max_bootstrapped_demos=3,
            max_labeled_demos=3,
        )
        
        # Compile the module
        optimized = optimizer.compile(
            module,
            trainset=trainset,
            trials=trials,
        )
        
        return optimized
    
    def generate_code_optimized(self, requirements: str) -> Dict[str, Any]:
        """
        Generate code using optimized prompts.
        
        Args:
            requirements: Code requirements
            
        Returns:
            Generated code and reasoning
        """
        result = self.code_generator(requirements=requirements)
        
        # Parse structured output
        try:
            # Attempt to extract code from response
            code = self._extract_code(result.code)
            return {
                "code": code,
                "reasoning": result.rationale,
                "success": True
            }
        except:
            return {
                "code": result.code,
                "reasoning": result.rationale,
                "success": False,
                "error": "Could not parse structured code"
            }
    
    def _extract_code(self, response: str) -> str:
        """Extract code from response."""
        # Simple extraction - can be made more sophisticated
        if "```" in response:
            parts = response.split("```")
            if len(parts) >= 2:
                return parts[1].strip()
        return response


# ============== Combined Pipeline ==============

class StructuredGemmaPipeline:
    """
    Complete pipeline combining Pydantic, Instructor, and DSPy.
    """
    
    def __init__(self, model_path: str = "google/gemma-2b"):
        # Initialize base model
        self.gemma = GemmaModel(model_name=model_path)
        self.gemma.load_model()
        
        # Initialize structured components
        self.instructor = InstructorGemma(self.gemma)
        self.dspy_gemma = StructuredDSPyGemma(self.gemma)
        
    def process_code_request(self, request: str) -> Dict[str, Any]:
        """
        Process a code-related request with full structure.
        
        Args:
            request: User request
            
        Returns:
            Structured response with code, analysis, and tests
        """
        # Step 1: Break down the task
        task_breakdown = self.instructor.break_down_task(request)
        
        # Step 2: Generate code with DSPy
        code_result = self.dspy_gemma.generate_code_optimized(request)
        
        if code_result["success"]:
            # Step 3: Analyze the generated code
            analysis = self.instructor.analyze_code(code_result["code"])
            
            # Step 4: Generate tests
            tests = self.instructor.generate_tests(code_result["code"])
            
            return {
                "success": True,
                "task_breakdown": task_breakdown.dict(),
                "code": code_result["code"],
                "analysis": analysis.dict(),
                "tests": tests.dict(),
                "reasoning": code_result["reasoning"]
            }
        else:
            return {
                "success": False,
                "error": code_result["error"],
                "task_breakdown": task_breakdown.dict()
            }
    
    def smart_qa(self, question: str, optimize: bool = False) -> QAResponse:
        """
        Answer questions with optional DSPy optimization.
        
        Args:
            question: Question to answer
            optimize: Whether to use DSPy optimization
            
        Returns:
            Structured QA response
        """
        if optimize:
            # Use DSPy multi-step reasoning
            reasoning_result = self.dspy_gemma.reasoner(problem=question)
            
            # Convert to structured output
            return QAResponse(
                answer=reasoning_result.final_answer,
                confidence=0.85,  # Would need proper confidence estimation
                reasoning=" -> ".join(reasoning_result.reasoning_steps),
                related_topics=[]  # Would need entity extraction
            )
        else:
            # Direct structured output
            return self.instructor.answer_question(question)