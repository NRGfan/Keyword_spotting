import pandas as pd


WORDS = ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes']
NUM_KEYWORDS = len(WORDS)
SILENCE = '_silence_'
UNKNOWN = '_unknown_'
BACKGROUND_NOISE = 'bg_noise'


audio_list = "*\data_prep\audio_list.txt"
csv_train = "*\data_prep\train_clean.csv"
csv_test = "*\data_prep\test_clean.csv"

def process(txtfile, output_train, output_test):
  df = pd.read_csv(txtfile)
  print(df)
  num_data = len(df)
  class_num = []
  labels = []
  for i in range(num_data):
    filepath = df.iloc[i, 0]

    if "\\" in filepath:
      splitter = "\\"
    elif "/" in filepath:
      splitter = "/"
    audio_label = str(df.iloc[i, 0]).split(splitter)
    for i in audio_label:
      if i in WORDS:
        audio_label = i
        break
      elif i == SILENCE:
        audio_label = SILENCE
        break
      elif i == BACKGROUND_NOISE:
        audio_label = BACKGROUND_NOISE
        break
      else:
        audio_label = UNKNOWN
    if (audio_label in WORDS):
      class_number = WORDS.index(audio_label)
    elif audio_label == SILENCE or audio_label == BACKGROUND_NOISE:
      audio_label = SILENCE
      class_number = NUM_KEYWORDS  # either silence or noise
    else:
      audio_label = UNKNOWN
      class_number = NUM_KEYWORDS + 1  # other classes

    # append to list
    class_num.append(class_number)
    labels.append(audio_label)

  # insert pandas column
  df.insert(1, "class", class_num)
  df.insert(2, "label", labels)
  df = df[df["filepath"].str.contains("filepath") == False]

  #split the dataframe into train and test and randomise the order
  test_set = df.sample(frac=0.2)
  train_set = df.drop(test_set.index)
  train_set = train_set.sample(frac=1.0)
  train_set.to_csv(output_train, index=False)
  print("Exported to", output_train, len(train_set))

  test_set.to_csv(output_test, index=False)
  print("Exported to", output_test, len(test_set))


process(audio_list, csv_train, csv_test )


