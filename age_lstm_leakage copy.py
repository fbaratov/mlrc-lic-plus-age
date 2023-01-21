import pickle
import pandas as pd
import random
from collections import namedtuple


def gender_pickle_generator(cap_model):
  '''
    This function generates the captions for the specified model. It only concerns
    age. It could be merged, but for the purposes of readability it is better to 
    keep this way.
    Arguments
    ---------
    cap_model : str
        The captioning model in string format.
  '''
  directory_age = {'nic':  pickle.load(open('bias_data/Show-Tell/gender_val_st10_th10_cap_mw_entries.pkl', 'rb')),
             'sat':         pickle.load(open('bias_data/Show-Attend-Tell/gender_val_sat_cap_mw_entries.pkl', 'rb')),
             'fc' :         pickle.load(open('bias_data/Att2in_FC/gender_val_fc_cap_mw_entries.pkl', 'rb')),
             'att2in':      pickle.load(open('bias_data/Att2in_FC/gender_val_att2in_cap_mw_entries.pkl', 'rb')),
             'updn':        pickle.load(open('bias_data/UpDn/gender_val_updn_cap_mw_entries.pkl', 'rb')),
             'transformer': pickle.load(open('bias_data/Transformer/gender_val_transformer_cap_mw_entries.pkl', 'rb')),
             'oscar':       pickle.load(open('bias_data/Oscar/gender_val_cider_oscar_cap_mw_entries.pkl', 'rb')),
             'nic_plus':    pickle.load(open('bias_data/Woman-Snowboard/gender_val_baselineft_cap_mw_entries.pkl', 'rb')),
             'nic_equalizer': pickle.load(open('bias_data/Woman-Snowboard/gender_val_snowboard_cap_mw_entries.pkl', 'rb')),
             'human':         pickle.load(open('bias_data/Human_Ann/gender_obj_cap_mw_entries.pkl', 'rb'))                          
             }
  return directory_age[cap_model]
       
def race_pickle_generator(cap_model):
  '''
    This function generates the captions for the specified model. It only concerns
    race. It could be merged, but for the purposes of readability it is better to 
    keep this way.
    Arguments
    ---------
    cap_model : str
        The captioning model in string format.
  '''
  directory_race = {'nic': pickle.load(open('bias_data/Show-Tell/race_val_st10_cap_entries.pkl', 'rb')),
             'sat':         pickle.load(open('bias_data/Show-Attend-Tell/race_val_sat_cap_entries.pkl', 'rb')),
             'fc' :         pickle.load(open('bias_data/Att2in_FC/race_val_fc_cap_entries.pkl', 'rb')),
             'att2in':      pickle.load(open('bias_data/Att2in_FC/race_val_att2in_cap_entries.pkl', 'rb')),
             'updn':        pickle.load(open('bias_data/UpDn/race_val_updn_cap_entries.pkl', 'rb')),
             'transformer': pickle.load(open('bias_data/Transformer/race_val_transformer_cap_entries.pkl', 'rb')),
             'oscar':       pickle.load(open('bias_data/Oscar/race_val_cider_oscar_cap_entries.pkl', 'rb')),
             'nic_plus':    pickle.load(open('bias_data/Woman-Snowboard/race_val_baselineft_cap_entries.pkl', 'rb')),
             'nic_equalizer': pickle.load(open('bias_data/Woman-Snowboard/race_val_snowboard_cap_entries.pkl', 'rb')),
             'human':         pickle.load(open('bias_data/Human_Ann/race_val_obj_cap_entries.pkl', 'rb')),                         
             }
  return directory_race[cap_model]

def label_human_caption(caption_list, young_words, old_words):
    '''
    Label human caption list from one entry of human captions.
    This rule is applied if any of the caption in the list contains 
    any word in young_words and old_words.
    The logic rules are simple:
    If entry corresponds to both old and young it is probably young 
    as we include man and woman in the words
    If entry is only young then it is young
    If entry is only old then it is old.
    Arguments
    ---------
    caption_list : list[str]
        The list of captions from five different human annotators
    young_words : list[str]   
        The list of words that is specified as young
    old_words : list[str]
        The list of words that is specified as old. 
    '''
    # If any of the words from any of the caption in the old_words list
    exists_old = any(word in caption.split() for word in old_words for caption in caption_list)
    # If any of the words from any of the caption is in young_words list
    exists_young = any(word in caption.split() for word in young_words for caption in caption_list)
    if exists_old and exists_young:
      return "Young"
    if exists_old:
        return "Old"
    if exists_young:
        return "Young"
    if not exists_old and not exists_young:
        return "Unknown"

def label_human_annotations(captions, young_words, old_words):
  '''
  This functions labels the human annotations from the captions dictionary.
  The captions dictionary is from their data.
  The dictionary has 5 different captions for a unique image id.
  If a caption contains the young or old words, then we label it as such.
  Arguments
  ---------
  captions : list[dict]
      Captions from the mscoco dataset
  young_words : list[str]
      List of words representing the young words
  old_words : list[str]
      List of words representing the old words
  '''
  # Create a data frame from captions
  data_frame = pd.DataFrame.from_records(captions,index=[str(id) for id in range(len(captions))])
  # Label the ages using the label_human_caption function
  labelled_ages = [label_human_caption(list, young_words, old_words) for list in data_frame['caption_list']]
  data_frame.insert(2, "bb_age", labelled_ages, True)
  data_frame['img_id'] =data_frame['img_id'].astype(int)
  return data_frame

def match_labels(labelled_human_captions, generated_captions):
  '''
  This function maps the labelled human captions to generated captions
  So we have the labels for generated captions as well.
  Arguments
  ---------
  labelled_human_captions : list[dict]
      labelled human captions dictionary
  generated_captions : list[dict]
      generated captions from the specified model.
  '''
  generated_captions_data_frame = pd.DataFrame.from_records(generated_captions,index=[str(id) for id in range(len(generated_captions))])
  generated_captions_data_frame['img_id'] = generated_captions_data_frame['img_id'].astype(int)
  # Merge the datasets using the img_id column because it has one to one match.
  generated_captions_data_frame = generated_captions_data_frame.merge(labelled_human_captions[['img_id', 'bb_age']], on='img_id')
  return generated_captions_data_frame

def make_train_test_split(args, age_task_mw_entries):
    '''
    This is the function from the authors code that define train and test splits
    from the entries.

    Arguments
    ---------
    args : NamedTuple
        The parsed arguments
    age_task_entries : list[dict]
        The entries for age task. This is basically the labelled dataframe.
    '''
    old_entries, young_entries = [], []
    for _ , entry in age_task_mw_entries.iterrows():
        if entry['bb_age'] == 'Young':
            young_entries.append(entry)
        elif entry['bb_age'] == 'Old':
            old_entries.append(entry)
    #print(len(old_entries))
    each_test_sample_num = round(len(young_entries) * args.test_ratio)
    each_train_sample_num = len(young_entries) - each_test_sample_num

    old_train_entries = [old_entries.pop(random.randrange(len(old_entries))) for _ in range(each_train_sample_num)]
    young_train_entries = [young_entries.pop(random.randrange(len(young_entries))) for _ in range(each_train_sample_num)]
    old_test_entries = [old_entries.pop(random.randrange(len(old_entries))) for _ in range(each_test_sample_num)]
    young_test_entries = [young_entries.pop(random.randrange(len(young_entries))) for _ in range(each_test_sample_num)]
    d_train = old_train_entries + young_train_entries
    d_test = old_test_entries + young_test_entries
    random.shuffle(d_train)
    random.shuffle(d_test)
    print('#train : #test = ', len(d_train), len(d_test))

    return d_train, d_test


if __name__ == "__main__":
  young_words = ['kid', 'kids', 'child', 'children', 'young', 'boy', 'boys', 'little', 'baby', 'babies',
               'childhood', 'babyhood', 'toddler', 'adolescence', 'teenager', 'teenagers',
               'schoolboy', 'schoolgirl', 'youngster', 'infant', 'preschooler'
               'toddler', 'student']
  old_words = ['elder', 'man', 'men', 'woman', 'women', 'old', 'elders', 'elderly', 'grandma', 'grandpa',
             'mom', 'dad', 'father', 'ancient', 'elder', 'aged', 'senior',
             'grandparent', 'senior']
  data_frame_labelled = label_human_annotations(gender_pickle_generator('human'), young_words, old_words)
  # Matches the labels with nic generated captions.
  data_frame_matched = match_labels(data_frame_labelled,gender_pickle_generator('nic') )
  # Emulator for args. Nothing important here.
  m = namedtuple('args', ['balanced_data', 'test_ratio'])
  args = m(True, 0.1)
  # Make the train and test split.
  d_train, d_test = make_train_test_split(args, data_frame_labelled)
  print("First element of train is: {}".format(d_train[0]))
  print("First element of test is: {}".format(d_test[0]))
