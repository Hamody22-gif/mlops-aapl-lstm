import requests
import json
import random

def test_prediction_endpoint():
    # The URL of our API (Localhost)
    url = "http://127.0.0.1:8000/predict"

    print(f"\n--- Testing API Endpoint: {url} ---")

    # 1. Generate Dummy Data
    # Let's simulate 60 days of stock prices (e.g., slowly rising from $150)
    # This mimics what a real data source would provide
    dummy_prices = [150.0 + (i * 0.2) + random.uniform(-1, 1) for i in range(60)]
    
    payload = {
        "features": dummy_prices
    }

    print(f"Sending {len(dummy_prices)} days of price data...")
    print(f"First 5 prices: {dummy_prices[:5]}...")

    try:
        # 2. Send POST Request
        response = requests.post(url, json=payload)
        
        # 3. Check Response
        if response.status_code == 200:
            result = response.json()
            print("\n✅ Success! The API responded.")
            print("---------------------------------------------------")
            print(json.dumps(result, indent=2))
            print("---------------------------------------------------")
            
            # Verify the math looks mostly sanity-checked
            pred = result.get("prediction_price_usd")
            print(f"Predicted Price: ${pred:.2f}")
            latest_price = dummy_prices[-1]
            print(f"Latest Input Price: ${latest_price:.2f}")
            
            diff = pred - latest_price
            print(f" Difference: ${diff:.2f}")
            
        else:
            print(f"\n❌ Error: Server returned status {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("\n❌ Connection Failed!")
        print("Is the API server running? (Did you double-click run_api.bat?)")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")

if __name__ == "__main__":
    test_prediction_endpoint()
