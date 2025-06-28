#!/usr/bin/env python3
"""
Setup HuggingFace authentication for Gemma models
"""

import os
import sys
from huggingface_hub import login, HfApi

def setup_huggingface():
    print("🤗 HuggingFace Authentication Setup")
    print("=" * 40)
    print("\nTo use Gemma models, you need:")
    print("1. A HuggingFace account")
    print("2. Access granted to google/gemma models")
    print("3. A HuggingFace access token")
    print("\nSteps:")
    print("1. Go to https://huggingface.co/google/gemma-2b")
    print("2. Click 'Agree and access repository'")
    print("3. Get your token from https://huggingface.co/settings/tokens")
    print("   (Create a new token with 'read' permissions if needed)")
    
    token = input("\nEnter your HuggingFace token (or 'skip' to set later): ").strip()
    
    if token.lower() != 'skip':
        try:
            login(token=token, add_to_git_credential=True)
            print("\n✅ Successfully logged in to HuggingFace!")
            
            # Test access
            api = HfApi()
            try:
                api.model_info("google/gemma-2b", token=token)
                print("✅ Confirmed access to Gemma models!")
            except Exception as e:
                print("⚠️  Could not verify Gemma access. Make sure you've accepted the license.")
                print(f"   Error: {e}")
        except Exception as e:
            print(f"\n❌ Login failed: {e}")
            sys.exit(1)
    else:
        print("\n⚠️  Skipped authentication. You can set it later with:")
        print("   huggingface-cli login")
        print("   or")
        print("   export HF_TOKEN='your-token-here'")

if __name__ == "__main__":
    setup_huggingface()