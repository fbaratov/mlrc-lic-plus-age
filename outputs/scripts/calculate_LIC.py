import numpy as np
from typing import Tuple,List,Dict
from dataclasses import dataclass
from parse_LIC import LIC_Calculator

@dataclass
class Statistics:
    '''
    This class represent the statistics defined in the paper. Just mean and standard deviation.
    '''
    mean: float       # 'a' has no default value
    std: float   # assign a default value for 'b'

def calculate_statistics(data: LIC_Calculator) -> Dict[str,Statistics]:
  '''
  Calculate the statistics from the LIC_Calculator class defined. This function
  returns a dictionary, which maps each model to statistics.
  Attributes
  ----------
  data : LIC_Calculator
      LIC_Calculator class for getting the models lic scores from the outputs
  Returns
  -------
  Dict[str,Statistics]
      Dictionary that maps each model 'str' into the statistics
  
  '''
  statistics = {}
  for index,model in enumerate(data.model_names.keys()):
    arr = np.array(data.LIC_scores[index])
    statistics[model] = Statistics(mean = arr.mean(), std = arr.std())
  return statistics

def calculate_LIC(data_human: LIC_Calculator, data_generated: LIC_Calculator) -> Dict[str,float]:
  '''
  Calculate the LIC from human and generated annotations defined. This function
  returns a dictionary, which maps each model to LIC.
  Attributes
  ----------
  data_human : LIC_Calculator
      LIC_Calculator class for getting the models lic scores from the outputs of the human
      annotations.
  data_generated : LIC_Calculator
      LIC_Calculator class for getting the models lic scores from the outputs of the generated
      annotations.
  Returns
  -------
  Dict[str,Statistics]
      Dictionary that maps each model 'str' into the LIC scores.
  
  '''
  LIC = {}
  assert data_human.model_names.keys() ==  data_generated.model_names.keys(), 'The data models do not contain the same models.'
  for index,model in enumerate(data_human.model_names.keys()):
    lic_M = np.array(data_generated.LIC_scores[index]).mean()
    lic_D = np.array(data_human.LIC_scores[index]).mean()
    #Definition of the LIC from the paper. We substract the model's lic from the data's lic, to find how much bias is amplified.     
    LIC[model] = lic_M - lic_D
  return LIC

if __name__ == "__main__":

  models = ['lstm', 'bert', 'pretrained_bert']
  annotations = ['generated', 'gender']
  classification_type = ['gender','race']
  for type in classification_type:
    for model in models:
      parsed_generated = LIC_Calculator.from_file(classifier_name = model,
                                      classification_type = type,
                                             annotation_type = "generated")
      parsed_human = LIC_Calculator.from_file(classifier_name = model,
                classification_type = type,
                annotation_type = "human")
      print("Calculating statistics for classification_type= {}, model= {}".format(type,model))
      print("Mean LIC scores with STD for generated, so LIC_M is:\n {}".format(calculate_statistics(parsed_generated)))
      print("Mean LIC scores with STD for human, so LIC_D is:\n {}".format(calculate_statistics(parsed_human)))
      print("LIC scores are \n {}".format(calculate_LIC(parsed_human, parsed_generated)))
