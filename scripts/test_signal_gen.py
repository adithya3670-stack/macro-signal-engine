from backtesting.signal_generator import SignalGenerator
import pandas as pd
import traceback

def test_gen():
    try:
        gen = SignalGenerator()
        print("SignalGenerator initialized.")
        
        assets = ['SP500', 'Bitcoin']
        print(f"Testing generation for: {assets}")
        
        signals = gen.generate_signals(assets, start_date='2020-01-01')
        print("Generation Complete.")
        print("Signals Shape:", signals.shape)
        print("Head:\n", signals.head())
    
        # Validation
        if signals['Bitcoin'].isnull().all():
            print("FAILURE: Bitcoin signals are all NaN!")
        else:
            print("SUCCESS: Bitcoin signals generated.")
            print("Bitcoin Non-NaN Count:", signals['Bitcoin'].count())
        
    except Exception as e:
        print("CRASHED:")
        print(e)
        traceback.print_exc()

if __name__ == "__main__":
    test_gen()
