# Extract important features and activation values

## Find important features

# Need to update your model layers
layers=["List of your model layers"] # e.g. layers=['conv1','conv2','conv3','conv4','fcn5','fcn6','fcn7']

# Choose a batch of 100 samples from input_image_4D
batch_size = 100

for c in range(num_classes)
  c = str(c)
  input_image_4D, num_samples = loadtrain(c)
  model=loadmodel()
  # Slice input_image_4D to get the batch
  for start_index in range(0, num_samples, batch_size):
    part = 1
    end_index = min(start_index + batch_size, num_samples)
    input_image_4D_batch = input_image_4D[start_index:end_index]

    for layer in layers:
      layer_name = getattr(model, layer)
      explainer = shap.GradientExplainer((model,layer_name), data=background_images, batch_size=1)
      Grad_shap_values, Grad_indexes = explainer.shap_values(input_image_4D_batch, ranked_outputs=1)
      with open('Results/Model-'+c+'/Gradient/batches/Gradient_shap_values_model_'+c+'_'+layer+'_batch-'+str(part)+'.pkl', 'wb') as file:
        pickle.dump(Grad_shap_values, file)
      torch.save(Grad_indexes, 'Results/Model-'+c+'/Gradient/batches/Gradient_index_values_model_'+c+'_'+layer+'_batch-'+str(part)+'.pt')
      del explainer,Grad_shap_values
      print('end',layer)
    part +=1
  del model

  #Merge SHAP result
  count=0
  for _, _, files in os.walk('Results/Model-'+c+'/Gradient/batches'):
    count += len(files)
  batch_no = int(count/(2*len(layers)))
  for layer in layers:
    Grad_shap = []
    Grad_index = []
    for batch in range(1,batch_no+1):
      with open('Results/Model-'+c+'/Gradient/batches/Gradient_shap_values_model_'+c+'_'+layer+'_batch-'+str(batch)+'.pkl', 'rb') as file:
          shap = pickle.load(file)
          Grad_shap.append(shap[0])

      index = torch.load('Results/Model-'+c+'/Gradient/batches/Gradient_index_values_model_'+c+'_'+layer+'_batch-'+str(batch)+'.pt')
      Grad_index.append(index)

    # Cobining batches
    combined_shap = np.vstack(Grad_shap)
    combined_index = torch.cat(Grad_index,dim=0)
    Incorrect_indices = np.nonzero(combined_index!= int(c)-1)
    if len(Incorrect_indices)>0:
      print("Incorrect indices in model",c)
    np.save('Results/Model-'+c+'/Gradient/shap_'+layer+'.npy', combined_shap)
    print(layer,np.shape(combined_shap))
  del shap,Grad_shap,Grad_index,combined_shap,combined_index

    
## Extract activation for each layer across all training images
### Extract dictionary
def get_activation(name):
    def hook(model, input, output):
        activation_All[name] = input[0].detach()
    return hook


for c in range(num_classes)

  input_image_4D, num_samples = loadtrain(c)

  for start_index in range(0, num_samples, batch_size):
    if 'activation_All' in globals():
      del activation_All
      activation_All = {}
    part = 1
    end_index = min(start_index + batch_size, num_samples)
    input_image_4D_batch = input_image_4D[start_index:end_index]

    for i in range(input_image_4D_batch.shape[0]):
      model=loadmodel()
      for name, layer in model.named_modules():
        final_name= str(i+start_index) + "_" + name
        layer.register_forward_hook(get_activation(final_name))
      output_of_layer = model(input_image_4D_batch[i].unsqueeze(0))

    # torch.save(activation_All, "Results/Model-"+c+"/Activation/Activation_part"+str(part)+".pt")
    print("End part:",part)

  # Extract activation for each layer across all training images

    for i,layer in enumerate(layers):
      Feature_map_values = [value.cpu().numpy() for key, value in activation_All.items() if key.endswith(layer)]
      Feature_map_values = np.array(Feature_map_values)
      # If the layers are convolutions (we have had 5 convolution layers)
      if 0<=i<=6:
        Feature_map_values = Feature_map_values.reshape((np.shape(Feature_map_values)[1],np.shape(Feature_map_values)[0],np.shape(Feature_map_values)[3],np.shape(Feature_map_values)[4],np.shape(Feature_map_values)[2]))
      else:
        Feature_map_values = Feature_map_values.reshape((np.shape(Feature_map_values)[1],np.shape(Feature_map_values)[0],np.shape(Feature_map_values)[2]))
      np.save('Results/Model-'+c+'/Activation/batches/Activation_'+layer+'_part'+str(part)+'.npy', Feature_map_values)
      print(layer,np.shape(Feature_map_values))
      del Feature_map_values

  del activation_All output_of_layer, input_image_4D_batch
