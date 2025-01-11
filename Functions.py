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
