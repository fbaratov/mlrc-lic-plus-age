from typing import List,Set,Dict
from collections import Counter
import re


def find_model_names(text : str) -> Dict[str, int]:
  '''
  This functions find all the model names in the text string. 
  The model names are expected to be the model names introduced in the paper
  so 'nic' 'transformer' etc ...
  Arguments
  ---------
  text : str
      Text extracted from the output text file for specific model.
  Returns
  -------
  Set[str]
      The set containing the model names and how many times the model was encountered.
  '''
  # Define the regex to match the group after the --cap_model. This is our caption model.
  match_string = r'--cap_model ([a-zA-Z0-9]+_[a-zA-Z0-9]+|[a-zA-Z0-9]+)'
  model_names = Counter()
  for matches in re.finditer(match_string, text):
    name = matches.group(1)
    model_names[name] += 1
  return model_names


def find_seeds(text: str) -> List[List[int]]:
  '''
  This functions find all the seeds in the text string. 
  Arguments
  ---------
  text : str
      Text extracted from the output text file for specific model.
  Returns
  -------
  List[List[int]]
      The list of lists containing the seeds for each model.
  '''
  # First find the model names.
  model_names = find_model_names(text).keys()
  match_string = r'--seed (\d+)(?=.*{} )'
  seeds = []
  for model in model_names:
    # Do lookup regular expression with the model name.
    # So this ensures that we are extracting the seeds for the specified model
    # in the model name.
    match_string_ = match_string.format(model)
    seed = []
    for matches in re.finditer(match_string_, text):
      seed.append(int(matches.group(1)))
    seeds.append(seed)
  return seeds

def find_lic_scores(text: str) -> List[List[float]]:
  '''
  This functions find all LIC scores in the text.
  Arguments
  ---------
  text : str
      Text extracted from the output text file for specific model.
  Returns
  -------
  List[List[int]]
      The list of lists containing the LIC scores for each model.
  ''' 
  # Find all the model names.
  model_names = find_model_names(text).keys()
  # This string extracts the section of the text that belongs to the model.
  match_string = r"--cap_model {} .*?#############################"
  # This string extracts the LIC score.
  score_string = r"LIC score .*: (\d+\.\d+)%"
  lic_scores = []
  for model in model_names:
    matches = re.findall(match_string.format(model), text, re.DOTALL)
    lic_score = []
    # Extract the section of the text given the model.
    for match in matches:
      for scores in re.finditer(score_string, match):
        # From this section, extract the LIC score.
        lic_score.append(float(scores.group(1)))
    lic_scores.append(lic_score)
  return lic_scores
  

def find_epochs(text: str) -> List[List[int]]:
  '''
  This functions find all epochd in the text.
  Arguments
  ---------
  text : str
      Text extracted from the output text file for specific model.
  Returns
  -------
  List[List[int]]
      The list of lists containing the epochs for each model.
  ''' 
  # Find all the model names.
  model_names = find_model_names(text).keys()
  # This string extract the epochs given the model name.
  match_string = r'--num_epochs (\d+)(?=.*{} )'
  epochs = []
  for model in model_names:
    # Match the epochs for the specified model.
    match_string_ = match_string.format(model)
    epoch = []
    # For the match in the match_string, append it to the epoch array.
    for matches in re.finditer(match_string_, text):
      epoch.append(int(matches.group(1)))
    epochs.append(epoch)
  return epochs
