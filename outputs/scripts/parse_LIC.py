from dataclasses import dataclass
from typing import List,Dict,Optional
from regex_helper import find_model_names,find_seeds,find_lic_scores,find_epochs


# Dataclass is frozen for immutability purposes. Maybe we will need to use this again,
# So make sure that no-one can change the attributes as they are important for other parts of program.
@dataclass(frozen=True)
class LIC_Calculator:
    '''
    This class represent the numbers we got from running the script.

    Attributes
    ----------
    LIC_scores : List[List[float]]
        The LIC scores from the script. Each sub-list contains for
        one model.
    model_names : Dict[str,int]
        Name of the models we are trying to find LIC score, and how many
        time they were encountered. It is expected to be 10. This does not
        refer to the name of the model we are using to classify, but to 
        generate. So 'nic', 'transformer' etc.
    classifier_name : str
        The name of the classifier we are using. This refers to the classification
        model. So 'lstm', 'bert' or 'pre-trained-bert'.
    classification_type : str
        The type of classification we are trying to calculate LIC score. 
        This is either 'gender' or 'race'
    annotation_type : str
        The type of annotation we are using. 
        This can be either 'human' or 'generated'
    seeds : List[List[int]]
        Seeds used for model to generate the LIC scores.
    nr_of_epochs : List[List[int]]
        How epochs this model was trained
      
    Methods
    -------
    from_file(model_name:str, classifer_name:str, classification_type:str)
        Find the human_score and generated_score for the model
    '''
    
    LIC_scores: List[List[float]]
    model_names : Dict[str,int]
    classifier_name : str
    classification_type : str
    annotation_type : str
    seeds : List[List[int]]
    nr_of_epochs: List[List[int]]
    @classmethod
    def from_file(cls,classifier_name:str, classification_type:str, annotation_type:str, file_name: Optional[str]):
      '''
      This function initializes the class for  classifier and classification_type, annotation_type
      Parameters
      ----------
      classifer_name : str
          This is the name of the model used for classifying the outputs. It is expected that
          this is either 'lstm', 'bert' or 'pretrained_bert'
      classification_type : str
          This is the type of classification we are doing. It can have two values.
          'gender' or 'race'
      annotation_type : str
          This is the annotation used. Either 'human' or 'generated'
      file_name : Optional[str]
          This is the file_name for reading the lic scores, usally we skip this
          but for testing purposes it can be used.
      Returns
      -------
      Self
          Class containing the attributes.
      '''
      if not file_name:
        file_name = "outputs/{}_{}_{}.txt".format(classifier_name, annotation_type, classification_type)
      #Search for file in the directory.
      with open(file_name, "r") as file:
        text = file.read()
        nr_of_epochs = find_epochs(text)
        seeds = find_seeds(text)
        model_names = find_model_names(text)
        LIC_scores = find_lic_scores(text)
      return cls(LIC_scores,
                 model_names,
                 classifier_name,
                 classification_type,
                 annotation_type,
                 seeds,
                 nr_of_epochs
                 )
