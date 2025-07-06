#!/usr/bin/env python3
"""
Test Neo4j connections with different URI schemes.
"""

import asyncio
import os
from neo4j import AsyncGraphDatabase

async def test_connection(uri, username, password):
    """Test connection to Neo4j with given credentials."""
    print(f"Testing connection to {uri} with user '{username}' and password '{password[:10]}...'")
    
    try:
        driver = AsyncGraphDatabase.driver(uri, auth=(username, password))
        async with driver.session() as session:
            result = await session.run("RETURN 1 as test")
            record = await result.single()
            print(f"✅ Connection successful! Test result: {record['test']}")
            return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False
    finally:
        if 'driver' in locals():
            await driver.close()

async def main():
    print("🔍 Testing Neo4j Connection")
    print("=" * 40)
    
    username = "neo4j"
    password = "Everythingisconnected"
    
    # Test different URI schemes
    uris = [
        "bolt://127.0.0.1:7687",
        "bolt://localhost:7687", 
        "neo4j://127.0.0.1:7687",
        "neo4j://localhost:7687"
    ]
    
    for uri in uris:
        await test_connection(uri, username, password)
        print()

if __name__ == "__main__":
    asyncio.run(main()) 