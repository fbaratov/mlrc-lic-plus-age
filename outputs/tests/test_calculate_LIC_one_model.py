import unittest
# Append the path where we can find the scripts.
import sys
sys.path.append('scripts')
from parse_LIC import LIC_Calculator

class TestLICParser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        file_name = "tests/test_files/test_file_1.txt"
        no_arguments = {"classifier_name" : None,
                            "classification_type" : None, 
                            "annotation_type" : None}
        cls.parsed =  LIC_Calculator.from_file(**no_arguments, file_name = file_name)


    def test_model_parsing(self):
        self.assertEqual(self.parsed.model_names.keys(), {'nic'})

    def test_epochs_parsing(self):
        self.assertEqual(self.parsed.nr_of_epochs, [[20,20]])


    def test_seeds_parsing(self):
        self.assertEqual(self.parsed.seeds, [[0,12]])
        
    
    def test_LIC_parsing(self):
        self.assertEqual(self.parsed.LIC_scores, [[41.75,45.44]])

    
if __name__ == '__main__':
    unittest.main()