#!/usr/bin/env python3
"""
Production-Ready API Server with Structured Gemma Outputs

This demonstrates how to build a reliable API service using
Pydantic + Instructor + DSPy for guaranteed structured responses.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
import uvicorn
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.structured.structured_gemma import (
    StructuredGemmaPipeline,
    CodeAnalysis,
    TaskBreakdown,
    QAResponse,
    TestSuite,
    DebugAnalysis
)

# Initialize FastAPI app
app = FastAPI(
    title="Gemma Structured API",
    description="Production-ready API with guaranteed structured outputs",
    version="1.0.0"
)

# Initialize pipeline (in production, do this once on startup)
pipeline = None

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============== Request/Response Models ==============

class CodeAnalysisRequest(BaseModel):
    code: str = Field(description="Source code to analyze")
    language: Optional[str] = Field(None, description="Programming language")


class TaskBreakdownRequest(BaseModel):
    task: str = Field(description="Task description")
    context: Optional[str] = Field(None, description="Additional context")


class QARequest(BaseModel):
    question: str = Field(description="Question to answer")
    context: Optional[str] = Field(None, description="Context for the question")
    use_reasoning: bool = Field(False, description="Use multi-step reasoning")


class DebugRequest(BaseModel):
    code: str = Field(description="Code with error")
    error_message: str = Field(description="Error message")
    language: Optional[str] = Field(None, description="Programming language")


class TestGenerationRequest(BaseModel):
    function_code: str = Field(description="Function to generate tests for")
    test_framework: str = Field("pytest", description="Test framework to use")


class BatchRequest(BaseModel):
    requests: List[Dict[str, Any]] = Field(description="Batch of requests")
    request_type: str = Field(description="Type of requests (analyze/debug/test)")


class APIResponse(BaseModel):
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    processing_time_ms: Optional[float] = None


# ============== API Endpoints ==============

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup."""
    global pipeline
    logger.info("Initializing Gemma pipeline...")
    pipeline = StructuredGemmaPipeline("google/gemma-2b")
    logger.info("Pipeline ready!")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Gemma Structured API",
        "version": "1.0.0",
        "endpoints": [
            "/analyze-code",
            "/break-down-task",
            "/answer-question",
            "/debug-code",
            "/generate-tests",
            "/batch-process"
        ],
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": pipeline is not None,
        "timestamp": datetime.now()
    }


@app.post("/analyze-code", response_model=APIResponse)
async def analyze_code(request: CodeAnalysisRequest):
    """
    Analyze code and return structured analysis.
    
    Always returns valid JSON with guaranteed fields:
    - language, purpose, complexity, score, improvements, etc.
    """
    start_time = datetime.now()
    
    try:
        # Get structured analysis
        analysis = pipeline.instructor.analyze_code(request.code)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return APIResponse(
            success=True,
            data=analysis.dict(),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Code analysis failed: {e}")
        return APIResponse(
            success=False,
            error=str(e)
        )


@app.post("/break-down-task", response_model=APIResponse)
async def break_down_task(request: TaskBreakdownRequest):
    """
    Break down a task into structured steps.
    
    Returns:
    - task_type, difficulty, steps, prerequisites, etc.
    """
    start_time = datetime.now()
    
    try:
        breakdown = pipeline.instructor.break_down_task(request.task)
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return APIResponse(
            success=True,
            data=breakdown.dict(),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Task breakdown failed: {e}")
        return APIResponse(
            success=False,
            error=str(e)
        )


@app.post("/answer-question", response_model=APIResponse)
async def answer_question(request: QARequest):
    """
    Answer questions with structured responses.
    
    Returns:
    - answer, confidence, reasoning, sources, related_topics
    """
    start_time = datetime.now()
    
    try:
        if request.use_reasoning:
            # Use DSPy multi-step reasoning
            response = pipeline.smart_qa(request.question, optimize=True)
        else:
            # Direct structured response
            response = pipeline.instructor.answer_question(
                request.question,
                request.context
            )
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return APIResponse(
            success=True,
            data=response.dict(),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Question answering failed: {e}")
        return APIResponse(
            success=False,
            error=str(e)
        )


@app.post("/debug-code", response_model=APIResponse)
async def debug_code(request: DebugRequest):
    """
    Debug code with structured analysis.
    
    Returns:
    - error_type, root_cause, fix_suggestion, code_fix, prevention_tips
    """
    start_time = datetime.now()
    
    try:
        debug_info = pipeline.instructor.debug_code(
            request.code,
            request.error_message
        )
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return APIResponse(
            success=True,
            data=debug_info.dict(),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Debugging failed: {e}")
        return APIResponse(
            success=False,
            error=str(e)
        )


@app.post("/generate-tests", response_model=APIResponse)
async def generate_tests(request: TestGenerationRequest):
    """
    Generate test cases for a function.
    
    Returns:
    - test_cases, edge_cases_covered, test_strategy, coverage_estimate
    """
    start_time = datetime.now()
    
    try:
        test_suite = pipeline.instructor.generate_tests(request.function_code)
        
        # Generate actual test code
        test_code = generate_test_code(test_suite, request.test_framework)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return APIResponse(
            success=True,
            data={
                **test_suite.dict(),
                "generated_code": test_code
            },
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Test generation failed: {e}")
        return APIResponse(
            success=False,
            error=str(e)
        )


@app.post("/batch-process", response_model=APIResponse)
async def batch_process(request: BatchRequest, background_tasks: BackgroundTasks):
    """
    Process multiple requests in batch.
    
    Useful for analyzing entire codebases or multiple tasks.
    """
    start_time = datetime.now()
    
    try:
        results = []
        
        for item in request.requests:
            if request.request_type == "analyze":
                result = pipeline.instructor.analyze_code(item["code"])
            elif request.request_type == "debug":
                result = pipeline.instructor.debug_code(
                    item["code"],
                    item["error_message"]
                )
            elif request.request_type == "test":
                result = pipeline.instructor.generate_tests(item["function_code"])
            else:
                raise ValueError(f"Unknown request type: {request.request_type}")
            
            results.append(result.dict())
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return APIResponse(
            success=True,
            data={
                "results": results,
                "total_processed": len(results)
            },
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        return APIResponse(
            success=False,
            error=str(e)
        )


@app.post("/complete-pipeline", response_model=APIResponse)
async def complete_pipeline(request: CodeAnalysisRequest):
    """
    Run complete pipeline: task breakdown -> code generation -> analysis -> tests.
    
    Demonstrates the full power of structured outputs.
    """
    start_time = datetime.now()
    
    try:
        # Process through complete pipeline
        result = pipeline.process_code_request(request.code)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return APIResponse(
            success=result["success"],
            data=result if result["success"] else None,
            error=result.get("error"),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return APIResponse(
            success=False,
            error=str(e)
        )


# ============== Helper Functions ==============

def generate_test_code(test_suite: TestSuite, framework: str = "pytest") -> str:
    """Generate actual test code from test suite."""
    if framework == "pytest":
        code = f"import pytest\nfrom mymodule import {test_suite.function_name}\n\n"
        
        for i, test in enumerate(test_suite.test_cases):
            code += f"def test_{test_suite.function_name}_{i+1}():\n"
            if test.description:
                code += f'    """{test.description}"""\n'
            code += f"    assert {test_suite.function_name}({test.input}) == {test.expected_output}\n\n"
        
        return code
    
    elif framework == "unittest":
        code = f"import unittest\nfrom mymodule import {test_suite.function_name}\n\n"
        code += f"class Test{test_suite.function_name.title()}(unittest.TestCase):\n\n"
        
        for i, test in enumerate(test_suite.test_cases):
            code += f"    def test_{test_suite.function_name}_{i+1}(self):\n"
            if test.description:
                code += f'        """{test.description}"""\n'
            code += f"        self.assertEqual({test_suite.function_name}({test.input}), {test.expected_output})\n\n"
        
        return code
    
    else:
        return "# Unsupported test framework"


# ============== Error Handlers ==============

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle all uncaught exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "details": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


# ============== Main ==============

if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )