# Finding Traces
# Saving SHAP of all layers
layers_conv = ["List of your model convolution layers"] # e.g. layers=['conv1','conv2','conv3','conv4']

def N_top_func(layer,c):

  percent = 0.1
  positive = 0
  N_tops = 0
  temp = [] # to save and find the min N top among all N tops of training images

  # load SHAPs values for a layer across all models
  shap_layer = np.load('Results/Model-'+c+'/Gradient/shap_'+layer+'.npy')

# Mode: without Averaging
  if shap_type=='none':
    if layer in layers_conv:
      for image in range(shap_layer.shape[0]):
        for matrix in shap_layer[image]:
          for row in matrix:
            for item in row:
              if item > 0:
                positive += 1
        temp.append(positive)
        positive=0

    else:
      for image in range(shap_layer.shape[0]):
        for item in shap_layer[image]:
          if item > 0:
            positive += 1
        temp.append(positive)
        positive=0

    N_tops= np.min(temp)

# Mode: Averaging over channels
  elif shap_type=='ch':
    if layer in layers_conv:
      shap_layer = np.mean(shap_layer, axis=1)
      for image in range(shap_layer.shape[0]):
        for row in shap_layer[image]:
          for item in row:
              if item > 0:
                  positive += 1
        temp.append(positive)
        positive=0
    else:
      for image in range(shap_layer.shape[0]):
        for item in shap_layer[image]:
          if item > 0:
            positive += 1
        temp.append(positive)
        positive=0

    N_tops = np.min(temp)


  print("The shape of the extracted shap:", np.shape(shap_layer))
  print("The minimum number of positive values existing in all the training images' SHAP:", N_tops)

  # Clear all variables except 'except_var'
  local_vars = list(locals().keys())
  except_var = ['N_tops','shap_layer']

  for var_name in local_vars:
      if var_name not in except_var:
          del locals()[var_name]

  return N_tops, shap_layer

##################################################

# Extracting Traces indices

# Finding top N SHAP indices
def N_top_indices(layer,N_top_count):

  shap_top_value = []
  #initial rate
  rate = 1
# Mode: without Averaging
  if shap_type=='none':
    if layer in layers_conv:
      temp1=[]
      for image in range(shap.shape[0]): # images numbers in the model
        flattened_list = np.array(shap[image]).flatten()
        top_indices_flat = np.argsort(-flattened_list)[:N_top]
        temp1.append(top_indices_flat)

      common_indices_flats = reduce(set.intersection, map(set, temp1))
      print(f"The numbers of common indices between all {shap.shape[0]} training images Step-1:",len(common_indices_flats))

      # Check if founded location are not enough
      if len(common_indices_flats) < N_top_count or len(common_indices_flats) < 10:
        rate=0.9
        c=2
        while len(common_indices_flats) < N_top_count or len(common_indices_flats) <10:
          all_numbers = [number for sublist in temp1 for number in sublist]
          number_counts = Counter(all_numbers)
          threshold = int((shap.shape[0])*rate)
          most_common_numbers = [number for number, count in number_counts.items() if count >= threshold]
          common_indices_flats = set(most_common_numbers)
          print(f"The numbers of common indices between {threshold} training images Step-{c}:",len(common_indices_flats))
          rate -= 0.05
          c += 1


      common_indices_flats = np.array(list(common_indices_flats))

      shape = np.array(shap[image]).shape

      for image in range(shap.shape[0]): # images numbers in the model
        ch, i, j = np.unravel_index(common_indices_flats, shape)
        shap_top_indexes = list(zip(ch, i, j))
        # find top values
        temp2=[]
        for idx in shap_top_indexes:
          ch, i, j = idx
          value = shap[image][ch][i][j]
          temp2.append(value)
        shap_top_value.append(temp2)

    else:
      common_indices_flats=[]
      temp1=[]
      for image in range(shap.shape[0]): # images numbers in the model
        indices = np.argsort(-shap[image])[:N_top]
        temp1.append(indices)

      common_indices = reduce(set.intersection, map(set, temp1))
      print(f"The numbers of common indices between all {shap.shape[0]} training images Step-1:",len(common_indices))

      # Check if founded location are not enough
      if len(common_indices) < N_top_count or len(common_indices) < 10:
        c=2
        while len(common_indices) < N_top_count  or len(common_indices) < 10:
          all_numbers = [number for sublist in temp1 for number in sublist]
          number_counts = Counter(all_numbers)
          threshold = int((shap.shape[0])*rate)
          most_common_numbers = [number for number, count in number_counts.items() if count >= threshold]
          common_indices = set(most_common_numbers)
          print(f"The numbers of common indices between {threshold} training images Step-{c}:",len(common_indices))
          rate -= 0.05
          c += 1


      shap_top_indexes = np.array(list(common_indices))

      for image in range(shap.shape[0]):
        # find top values
        temp2=[]
        for idx in shap_top_indexes:
          value = shap[image][idx]
          temp2.append(value)
        shap_top_value.append(temp2)

# Mode: Averaging over channels
  elif shap_type=='ch':
    if layer in layers_conv:
      temp1=[]
      for image in range(shap.shape[0]): # images numbers in the model
        flattened_list = np.array(shap[image]).flatten()
        top_indices_flat = np.argsort(-flattened_list)[:N_top]
        temp1.append(top_indices_flat)

      common_indices_flats = reduce(set.intersection, map(set, temp1))
      print(f"The numbers of common indices between all {shap.shape[0]} training images Step-1:",len(common_indices_flats))

      # Check if founded location are not enough
      if len(common_indices_flats) < N_top_count or len(common_indices_flats) < 10:
        c=2
        while len(common_indices_flats) < N_top_count or len(common_indices_flats) < 10:
          all_numbers = [number for sublist in temp1 for number in sublist]
          number_counts = Counter(all_numbers)
          threshold = int((shap.shape[0])*p)
          most_common_numbers = [number for number, count in number_counts.items() if count >= threshold]
          common_indices_flats = set(most_common_numbers)
          print(f"The numbers of common indices between {threshold} training images Step-{c}:",len(common_indices_flats))
          p -= 0.05
          c += 1

      common_indices_flats = np.array(list(common_indices_flats))

      shape = np.array(shap[image]).shape

      for image in range(shap.shape[0]):
        # Convert the 1D indices to (i, j) format
        i, j = np.unravel_index(common_indices_flats, shape)
        # Combine the (i, j) indices into a list of tuples
        shap_top_indexes = list(zip(i, j))
        # find top values
        temp2=[]
        for idx in shap_top_indexes:
          i, j = idx
          value = shap[image][i][j]
          temp2.append(value)
        shap_top_value.append(temp2)

    else:
      common_indices_flats=[]
      temp1=[]
      for image in range(shap.shape[0]): # images numbers in the model
        indices = np.argsort(-shap[image])[:N_top]
        temp1.append(indices)

      common_indices = reduce(set.intersection, map(set, temp1))
      print(f"The numbers of common indices between all {shap.shape[0]} training images Step-1:",len(common_indices))

      # Check if founded location are not enough
      if len(common_indices) < N_top_count  or len(common_indices) < 10:
        c=2
        while len(common_indices) < N_top_count or len(common_indices) <10:
          all_numbers = [number for sublist in temp1 for number in sublist]
          number_counts = Counter(all_numbers)
          threshold = int((shap.shape[0])*rate)
          most_common_numbers = [number for number, count in number_counts.items() if count >= threshold]
          common_indices = set(most_common_numbers)
          print(f"The numbers of common indices between {threshold} training images Step-{c}:",len(common_indices))
          rate -= 0.05
          c += 1


      shap_top_indexes = np.array(list(common_indices))

      for image in range(shap.shape[0]):
        # find top values
        temp2=[]
        for idx in shap_top_indexes:
          value = shap[image][idx]
          temp2.append(value)
        shap_top_value.append(temp2)


  print('The shap of the "top SHAP values":', np.shape(shap_top_value))

  # Clear all variables except 'except_var'
  local_vars = local_vars = list(locals().keys())
  except_vars = ['sorted_shap_top_indexes', 'shap_top_value', 'common_indices_flats']

  mean_shap_top_value = np.mean(shap_top_value,axis=0)
  sorted_data = sorted(zip(mean_shap_top_value, shap_top_indexes))
  sorted_mean_shap_top_value, sorted_shap_top_indexes = zip(*sorted_data)

  for var_name in local_vars:
      if var_name not in except_vars:
          del locals()[var_name]

  return sorted_shap_top_indexes, shap_top_value, common_indices_flats
