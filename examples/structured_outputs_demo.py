#!/usr/bin/env python3
"""
Demo: Structured Outputs with Gemma using Pydantic, Instructor, and DSPy

This demo shows how to build reliable, production-ready applications
with guaranteed structured outputs and self-optimizing prompts.
"""

import json
from typing import List, Dict
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.structured.structured_gemma import (
    StructuredGemmaPipeline,
    CodeAnalysis,
    TaskBreakdown,
    QAResponse,
    TestSuite,
    Difficulty
)


def demo_code_analysis():
    """Demo: Analyze code with structured output."""
    print("=" * 60)
    print("DEMO 1: Code Analysis with Guaranteed Structure")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = StructuredGemmaPipeline("google/gemma-2b")
    
    # Sample code to analyze
    code = """
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1
"""
    
    # Get structured analysis
    print("\nAnalyzing code...")
    analysis = pipeline.instructor.analyze_code(code)
    
    # Display results - guaranteed to have all fields!
    print(f"\n📊 Analysis Results:")
    print(f"Language: {analysis.language}")
    print(f"Purpose: {analysis.purpose}")
    print(f"Time Complexity: {analysis.complexity}")
    print(f"Space Complexity: {analysis.space_complexity}")
    print(f"Score: {analysis.score}/10")
    
    print("\n✅ Strengths:")
    for strength in analysis.strengths:
        print(f"  - {strength}")
    
    print("\n💡 Improvements:")
    for improvement in analysis.improvements:
        print(f"  - {improvement}")
    
    # Export as JSON - always valid!
    with open("code_analysis.json", "w") as f:
        json.dump(analysis.dict(), f, indent=2)
    print("\n✅ Exported to code_analysis.json")


def demo_task_breakdown():
    """Demo: Break down complex tasks."""
    print("\n" + "=" * 60)
    print("DEMO 2: Task Breakdown with Structured Planning")
    print("=" * 60)
    
    pipeline = StructuredGemmaPipeline("google/gemma-2b")
    
    task = "Build a REST API for a todo list application with authentication"
    
    print(f"\nTask: {task}")
    print("\nBreaking down task...")
    
    breakdown = pipeline.instructor.break_down_task(task)
    
    print(f"\n📋 Task Analysis:")
    print(f"Type: {breakdown.task_type}")
    print(f"Difficulty: {breakdown.difficulty.value}")
    print(f"Estimated Time: {breakdown.estimated_time}")
    
    print("\n📚 Prerequisites:")
    for prereq in breakdown.prerequisites:
        print(f"  - {prereq}")
    
    print("\n📝 Steps:")
    for i, step in enumerate(breakdown.steps, 1):
        print(f"  {i}. {step}")
    
    print("\n⚠️  Potential Challenges:")
    for challenge in breakdown.potential_challenges:
        print(f"  - {challenge}")
    
    print("\n✅ Success Criteria:")
    for criterion in breakdown.success_criteria:
        print(f"  - {criterion}")


def demo_test_generation():
    """Demo: Generate comprehensive test suites."""
    print("\n" + "=" * 60)
    print("DEMO 3: Automatic Test Generation")
    print("=" * 60)
    
    pipeline = StructuredGemmaPipeline("google/gemma-2b")
    
    function_code = """
def validate_email(email):
    '''Validate if an email address is properly formatted.'''
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))
"""
    
    print("\nGenerating test suite...")
    test_suite = pipeline.instructor.generate_tests(function_code)
    
    print(f"\n🧪 Test Suite for: {test_suite.function_name}")
    print(f"Strategy: {test_suite.test_strategy}")
    print(f"Coverage Estimate: {test_suite.coverage_estimate}%")
    
    print("\n📝 Test Cases:")
    for i, test in enumerate(test_suite.test_cases, 1):
        edge = " [EDGE CASE]" if test.edge_case else ""
        print(f"\n  Test {i}{edge}:")
        print(f"    Input: {test.input}")
        print(f"    Expected: {test.expected_output}")
        if test.description:
            print(f"    Description: {test.description}")
    
    print("\n✅ Edge Cases Covered:")
    for edge_case in test_suite.edge_cases_covered:
        print(f"  - {edge_case}")
    
    # Generate pytest code
    print("\n📄 Generated pytest code:")
    print("```python")
    print("import pytest")
    print(f"from mymodule import {test_suite.function_name}\n")
    
    for i, test in enumerate(test_suite.test_cases):
        print(f"def test_{test_suite.function_name}_{i+1}():")
        print(f"    assert {test_suite.function_name}({test.input!r}) == {test.expected_output}")
        print()
    print("```")


def demo_complete_pipeline():
    """Demo: Complete code generation pipeline."""
    print("\n" + "=" * 60)
    print("DEMO 4: Complete Code Generation Pipeline")
    print("=" * 60)
    
    pipeline = StructuredGemmaPipeline("google/gemma-2b")
    
    request = "Create a Python function to find all prime numbers up to n using the Sieve of Eratosthenes"
    
    print(f"\nRequest: {request}")
    print("\nProcessing complete pipeline...")
    
    result = pipeline.process_code_request(request)
    
    if result["success"]:
        print("\n✅ Pipeline completed successfully!")
        
        # Task breakdown
        breakdown = result["task_breakdown"]
        print(f"\n📋 Difficulty: {breakdown['difficulty']}")
        print(f"📋 Estimated Time: {breakdown['estimated_time']}")
        
        # Generated code
        print("\n💻 Generated Code:")
        print("```python")
        print(result["code"])
        print("```")
        
        # Code analysis
        analysis = result["analysis"]
        print(f"\n📊 Code Quality Score: {analysis['score']}/10")
        print(f"⚡ Complexity: {analysis['complexity']}")
        
        # Tests
        tests = result["tests"]
        print(f"\n🧪 Generated {len(tests['test_cases'])} test cases")
        print(f"📈 Coverage: {tests['coverage_estimate']}%")
        
        # Save complete result
        with open("pipeline_result.json", "w") as f:
            json.dump(result, f, indent=2)
        print("\n✅ Complete result saved to pipeline_result.json")
    else:
        print(f"\n❌ Pipeline failed: {result['error']}")


def demo_dspy_optimization():
    """Demo: DSPy prompt optimization."""
    print("\n" + "=" * 60)
    print("DEMO 5: Self-Optimizing Prompts with DSPy")
    print("=" * 60)
    
    pipeline = StructuredGemmaPipeline("google/gemma-2b")
    
    # Create training examples for DSPy
    print("\n🧠 Training DSPy for optimized code generation...")
    
    # Simulate optimization (in practice, you'd have real training data)
    requirements = "Function to calculate factorial"
    
    print(f"\nRequirements: {requirements}")
    print("\nGenerating with optimized prompts...")
    
    result = pipeline.dspy_gemma.generate_code_optimized(requirements)
    
    if result["success"]:
        print("\n💻 Optimized Code Generation:")
        print("```python")
        print(result["code"])
        print("```")
        
        print("\n🤔 Reasoning Chain:")
        print(result["reasoning"])
    else:
        print(f"\n❌ Generation failed: {result['error']}")


def demo_api_integration():
    """Demo: Building a reliable API with structured outputs."""
    print("\n" + "=" * 60)
    print("DEMO 6: Production-Ready API Example")
    print("=" * 60)
    
    # Simulate API endpoint
    from flask import Flask, jsonify, request
    
    print("\n🌐 Example API Endpoints:")
    
    print("""
@app.post("/api/analyze-code")
def analyze_code_endpoint():
    code = request.json["code"]
    analysis = pipeline.instructor.analyze_code(code)
    return jsonify(analysis.dict())  # Always returns valid JSON!

@app.post("/api/debug")
def debug_endpoint():
    data = request.json
    debug_info = pipeline.instructor.debug_code(
        data["code"], 
        data["error"]
    )
    return jsonify(debug_info.dict())  # Structured debugging info!

@app.post("/api/generate-tests")
def generate_tests_endpoint():
    code = request.json["code"]
    tests = pipeline.instructor.generate_tests(code)
    return jsonify(tests.dict())  # Complete test suite as JSON!
""")
    
    print("\n✅ Benefits:")
    print("  - 100% valid JSON responses")
    print("  - Type-safe with automatic validation")
    print("  - No parsing errors or malformed outputs")
    print("  - Ready for production deployment")


def main():
    """Run all demos."""
    print("🚀 Gemma Structured Output Demos")
    print("================================")
    print("\nThese demos show how Pydantic + Instructor + DSPy")
    print("turn Gemma into a production-ready AI system.")
    
    demos = [
        ("Code Analysis", demo_code_analysis),
        ("Task Breakdown", demo_task_breakdown),
        ("Test Generation", demo_test_generation),
        ("Complete Pipeline", demo_complete_pipeline),
        ("DSPy Optimization", demo_dspy_optimization),
        ("API Integration", demo_api_integration),
    ]
    
    for name, demo_func in demos:
        input(f"\n\nPress Enter to run: {name}")
        try:
            demo_func()
        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            print("(This is likely due to missing model authentication)")
    
    print("\n\n🎉 All demos completed!")
    print("\n📚 Key Takeaways:")
    print("  1. Pydantic ensures all outputs are structured and validated")
    print("  2. Instructor guarantees valid JSON from Gemma")
    print("  3. DSPy optimizes prompts automatically")
    print("  4. Combined = Production-ready AI applications")
    print("\n💡 Perfect for hackathon projects that need reliability!")


if __name__ == "__main__":
    main()