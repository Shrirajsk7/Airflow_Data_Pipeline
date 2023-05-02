import pandas as pd
import unittest
from feature_engineering import process_chunk

class TestFeatureEngineering(unittest.TestCase):
    
    def test_process_chunk(self):
        # creating sample data:
        sample_data = {'Volume': [357000, 56100], 'Adj Close': [20.12400055, 410]}
        df = pd.DataFrame(sample_data)

        # call the function to be tested
        result_df = process_chunk(df)
        # print(result_df)

        # Checking that new columns have been added to the processed data:
        self.assertIn('vol_moving_avg', result_df.columns)
        self.assertIn('adj_close_rolling_med', result_df.columns)

        print("Test passed: new columns were added to the processed data")


if __name__ == '__main__':
    unittest.main()