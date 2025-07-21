#!/usr/bin/env python3
"""
Test script to verify LanceDB integration between RAG client and agno agent.
This script will clear the database safely and test the integration.
"""

import os
import shutil
from lancedb_rag import RagClient
from agent.analysis_engine import lance_agent

def clean_database():
    """Safely remove the LanceDB directory."""
    db_path = "tmp/lancedb"
    if os.path.exists(db_path):
        print(f"Removing existing database at {db_path}")
        shutil.rmtree(db_path)
        print("Database cleared successfully")
    else:
        print("No existing database found")

def test_integration():
    """Test the integration between RAG client and agno agent."""
    print("=== Testing LanceDB Integration ===")
    
    # Step 1: Clean the database
    clean_database()
    
    # Step 2: Initialize RAG client and inject test data
    print("\nInitializing RAG client...")
    rag_client = RagClient()
    
    # Test text about the 12 rules
    test_transcript = """
    The 12 Grim Rules for a Perfect Life:
    
    1. Accept that life is suffering
    2. Take responsibility for your own life
    3. Make friends with people who want the best for you
    4. Compare yourself to who you were yesterday, not to who someone else is today
    5. Do not let your children do anything that makes you dislike them
    6. Set your house in perfect order before you criticize the world
    7. Pursue what is meaningful, not what is expedient
    8. Tell the truth - or at least don't lie
    9. Assume that the person you are listening to might know something you don't
    10. Be precise in your speech
    11. Do not bother children when they are skateboarding
    12. Pet a cat when you encounter one on the street
    
    These rules are about taking personal responsibility and finding meaning in life.
    """
    
    print("Injecting test transcript...")
    rag_client.inject_text(test_transcript, name="12 Grim Rules for a Perfect Life")
    
    # Step 3: Test agent access
    print("\nTesting agent access to injected data...")
    try:
        lance_agent("what are the 12 rules of life?")
        print("\n✅ Integration test completed successfully!")
    except Exception as e:
        print(f"❌ Agent failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_integration()
    if success:
        print("\n🎉 Integration working correctly!")
    else:
        print("\n💥 Integration test failed!")