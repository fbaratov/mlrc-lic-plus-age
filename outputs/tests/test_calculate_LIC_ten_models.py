import unittest
# Append the path where we can find the scripts.
import sys
sys.path.append('scripts')
from parse_LIC import LIC_Calculator

class TestLICParser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        file_name = "tests/test_files/test_file_2.txt"
        no_arguments = {"classifier_name" : None,
                            "classification_type" : None, 
                            "annotation_type" : None}
        cls.parsed =  LIC_Calculator.from_file(**no_arguments, file_name = file_name)


    def test_model_parsing(self):
        expected_result = {'nic','fc', 'att2in', 
                          'transformer', 'nic_plus', 'nic_equalizer',
                          'sat', 'oscar', 'updn'}
        self.assertEqual(self.parsed.model_names.keys(), expected_result )

    def test_epochs_parsing(self):
        expected_result = [[20 for _ in range(3)] for _ in range(9)]
        self.assertEqual(self.parsed.nr_of_epochs, expected_result)


    def test_seeds_parsing(self):
        expected_result = [[0,12,456] for _ in range(9)]
        self.assertEqual(self.parsed.seeds, expected_result)
        
    
    def test_LIC_parsing(self):
        expected_result = [[40.13,39.84,41.80],
                           [38.99, 39.56, 40.93],
                           [37.51,38.08,39.94],
                           [38.15, 38.75,39.12],
                           [39.40,39.08,40.21],
                           [39.43, 39.97, 41.72],
                           [39.28,39.84,40.45],
                           [38.99,39.74,40.47],
                           [39.48,39.72,40.26]]
        self.assertEqual(self.parsed.LIC_scores, expected_result)