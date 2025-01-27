# Generate a specific input image for SHAP explanations from "Test dataset"

def loadtest(idx):
  test_dir = 'Test_M_27'

  path_images = glob.glob(test_dir + '/D*.jpg')
  path_images = sorted(path_images)
  indices = idx
  input_images = [path_images[i] for i in indices]

  # Load one image as PIL as input image
  input_image_list = [Image.open(file_path) for file_path in input_images]

  #  Apply prnu and convert input image to tensor
  input_image = [prnu(image) for image in input_image_list]

  input_image_4D = torch.stack(input_image)   # or:  input_image_4D = input_image.unsqueeze(0)  # Add a new dimension to the tenso
  input_image_4D = input_image_4D.to(device)

  Ground_truth = []
  for f in input_images:
    image_path = f
    temp= image_path.split('/D')[1]
    number= temp.split('_')[0]
    Ground_truth.append(int(number))

  return input_image_4D, Ground_truth

###################################################################################

# Generate a specific input image for SHAP explanations from "Train dataset"
def loadtrain(model_no,idx1,idx2):

  path_images = glob.glob(f'Train_model_dataset/D{model_no}*.jpg')
  path_images = sorted(path_images)
  indices = list(range(idx1,idx2))

  input_images = [path_images[i] for i in indices]

  # Load one image as PIL as input image
  input_image_list = [Image.open(file_path) for file_path in input_images]

  # Apply prnu and convert input image to tensor
  input_image = [prnu(image) for image in input_image_list]
  # input_image = [transform(image) for image in input_image_list]

  input_image = torch.stack(input_image)
  input_image_4D = input_image.to(device)


  return input_image_4D



# Generate background images for SHAP explanations
background_images=[]
load_background_images = glob.glob('Explainable/Background/*.jpg')
load_background_images = sorted(load_background_images)
# Load the images as PIL
for i in range (0,len(load_background_images)):
  background_image = Image.open(load_background_images[i])
  background_image = prnu(background_image)
  background_images.append(background_image)

# # Convert the batch images to a tensor
background_images = torch.stack(background_images)
print(background_images.shape)

# Move to GPU
background_images = background_images.to(device)


############################################
# Calculate the std of all N top indices across all training images for all layers

def std_2():

  Std_top_value=[]
  shap = np.transpose(shap_top_values, (1, 0))

  for i in range(len(activation_top_values)):
    mask = shap[i] > 0
    indices = np.nonzero(mask)

    # Apply the mask to select items with positive SHAP
    new_activation_top_values = [activation_top_values[i][x] for x in indices[0]]
    std =  np.std(new_activation_top_values)
    Std_top_value.append(std)

  print("Standard Deviation of the top values of all training images:", np.shape(Std_top_value))

  return Std_top_value


####################################
# Save print() outputs

class OutputToFile:
    def __init__(self, filename):
        self.file = open(filename, 'w')
        self.stdout = sys.stdout

    def write(self, text):
        self.file.write(text)
        self.stdout.write(text)

    def flush(self):
        self.file.flush()
        self.stdout.flush()
########################################
# Save results

def save_result(data,dir):
  with open(dir, 'wb') as f:
      pickle.dump(data, f)
########################################
# Load results

def load_result(dir):
  with open(dir, 'rb') as f:
      data = pickle.load(f)

  return data

