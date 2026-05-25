import requests

def test_query():
    response = requests.get(
        "http://127.0.0.1:8000/ask",
        params={"question": "What do I love?"},
    )
    
    if response.status_code != 200:
        raise Exception(f"Server returned {response.status_code}: {response.text}")
    
    answer = response.json()["answer"]

    # Check for key concepts
    assert "dogs" in answer.lower(), "Missing 'dogs' keyword"
    #assert "eating" in answer.lower(), "Missing 'eating' keyword"

    print("✅ query test passed")

if __name__ == "__main__":
    test_query()
    print("All semantic tests passed!")
