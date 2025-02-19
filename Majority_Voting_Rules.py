import numpy as np
import pickle

From Functions import load_result


# Majority Voting
def MV():

  layers = model_layers
  Cmodels = class_name
  prc=0.05
  shap_type = 'ch'  # none,ch
  mode= 'Gradient'  # Gradient, Deep, Grad-CAM
  Method = 'Euclidean' # Euclidean , Static
  count=0
  path="Results/All_Outputs/Model-"
  For M in Cmodels:
    count = 0
    model_error=0
  
    test_path = f"new/Model-{M}/Test_Activation-{M}"
    path2= f"new/Model-{M}/Results/Euclidean/"
  
    cnt_modelA = load_result(path+f"{M}/Results/count_model-7A-less.pickle")
    prc_modelA = load_result(path+f"{M}/Results/percent_model-7A-less.pickle")
    cnt_modelB = load_result(path+f"{M}/Results/count_model-7B-less.pickle")
    prc_modelB = load_result(path+f"{M}/Results/percent_model-7B-less.pickle")
  
  
    with open(test_path+'/Ground_truth.pkl', 'rb') as file:
        ground_truth = pickle.load(file)
  
    with open(test_path+'/Prediction.pkl', 'rb') as file:
        pred = pickle.load(file)
  
  
  
    first_layerA = []
    last_layerA = []
    max_layerA = []
    max_count_A = []
  
    first_layerB = []
    last_layerB = []
    max_layerB = []
    max_count_B = []
  
    for i in range(len(ground_truth)):
      if ground_truth[i]!=pred[i]:
        model_error+=1
      layers_countA = [0 for _ in range(len(Cmodels))]
      print("===================================================")
      print("===================================================")
      print("Sample Test:",i+1)
      print("Ground Truth:",ground_truth[i],"-->",class_names[ground_truth[i]-1])
      print("Prediction:",pred[i],"-->",class_names[pred[i]-1])
      print("===================================================")
      print("===================================================")
      print("{:<6} {:<15} {:<12}".format("Layer", "Detected Model", "Accuracy"))
      for l, layer in enumerate(layers):
        max = 0
        detectA=''
        for m,model in enumerate(Cmodels):
          if prc_modelA[m][l][i][0] > max:
            max = prc_modelA[m][l][i][0]
            detectA= prc_modelA[m][l][i][1]
          elif prc_modelA[m][l][i][0] == max and max!=0:
            detectA= detectA + "," + prc_modelA[m][l][i][1]
  
        print("{:<12} {:<9} {}".format(layer, detectA, max))
        print("-----------------------------")
  
    # Find the winner model for each model
        numbers_detectA = detectA.split(',')
        numbers_detectA = [item for item in numbers_detectA if item != '']
  
        if layer=='conv1':
          first_layerA.insert(i,[int(item) for item in numbers_detectA])
        if layer=='fcn7':
          last_layerA.insert(i,[int(item) for item in numbers_detectA])
  
        for win in numbers_detectA:
          layers_countA[int(win)-1]=layers_countA[int(win)-1]+1
  
    # Print the winner model for the sample
      max_valueA = np.max(layers_countA)
      max_indicesA = [i for i, value in enumerate(layers_countA) if value == max_valueA] # find winner camera models
      outputA = [index + 1 for index in max_indicesA]  # sum 1 to match index with winners numbers
      max_layerA.insert(i,outputA)
      max_count_A.insert(i,[outputA,max_valueA])
      print("Max = Model(s)", outputA, "with:",max_valueA," votes")
  
  
    # Calculation for B
      layers_countB = [0 for _ in range(len(Cmodels))]
      print("===================================================")
      print("===================================================")
      print("Sample Test:",i+1)
      print("Ground Truth:",ground_truth[i],"-->",class_names[ground_truth[i]-1])
      print("Prediction:",pred[i],"-->",class_names[pred[i]-1])
      print("===================================================")
      print("===================================================")
      print("{:<6} {:<15} {:<12}".format("Layer", "Detected Model", "Accuracy"))
      for l,layer in enumerate(layers):
        max = 0
        detectB =''
        for m,model in enumerate(Cmodels):
          if prc_modelB[m][l][i][0] > max:
            max = prc_modelB[m][l][i][0]
            detectB= prc_modelB[m][l][i][1]
          elif prc_modelB[m][l][i][0] == max and max!=0:
            detectB= detectB + "," + prc_modelB[m][l][i][1]
  
        print("{:<12} {:<9} {}".format(layer, detectB, max))
        print("-----------------------------")
  
    # Find the winner model for each layer over all Models
        numbers_detectB = detectB.split(',')
        numbers_detectB = [item for item in numbers_detectB if item != '']
        if layer=='conv1':
          first_layerB.insert(i,[int(item) for item in numbers_detectB])
        if layer=='fcn7':
          last_layerB.insert(i,[int(item) for item in numbers_detectB])
        for win in numbers_detectB:
          layers_countB[int(win)-1]=layers_countB[int(win)-1]+1
  
  
    # Print the winner model for the sample
      max_valueB = np.max(layers_countB)
      max_indicesB = [i for i, value in enumerate(layers_countB) if value == max_valueB] # find winner camera models
      outputB = [index + 1 for index in max_indicesB]  # sum 1 to match index with winners numbers
      max_layerB.insert(i,outputB)
      max_count_B.insert(i,[outputB,max_valueB])
      print("Max = Model(s)", outputB, "with:",max_valueB," votes")



  ###############################################################

def Rules():

   # RuleExtraction
  
    for i in range(len(ground_truth)):
      count=count+1
      top_models_AB = {}
  
      for item in max_count_A[i][0]:
        if item not in top_models_AB.keys():
          top_models_AB[item] = max_count_A[i][1]
        else:
          top_models_AB[item] = top_models_AB[item]+max_count_A[i][1]
  
      for item in max_count_B[i][0]:
        if item not in top_models_AB.keys():
          top_models_AB[item] = max_count_B[i][1]
        else:
          top_models_AB[item] = top_models_AB[item]+max_count_B[i][1]
  
        top_camera_count = np.max(list(top_models_AB.values()))
        top_camera = [key for key, value in top_models_AB.items() if value == top_camera_count]
  
      print(max_count_A[i],max_count_B[i],top_camera,top_camera_count)
      # print(max_count_A[i],top_camera,top_camera_count)
  
  
    # Rule:1
      elif (pred[i] in top_camera) and top_camera_count >= model_threshold:
        print(count,"Rule 1, Confirm!",[i+1,top_camera,top_camera_count,pred[i]])
        multi_decision += 1
        if pred[i] == ground_truth[i]:
          TP+=1
          g_TP+=1
        else:
          FP+=1
          FP_result.append(["Rule 1","sample:",i+1,"Ground:",ground_truth[i],"Us:",top_camera,"Count:",top_camera_count,"Pred:",pred[i]])
          Wrong+=1
          g_FP+=1
          g_Wrong+=1
  
    # Rule:2
      elif (pred[i] not in top_camera) or top_camera_count < model_threshold :
        print(count,"Rule 2, Not detected!",[i+1,top_camera,top_camera_count,pred[i]])
        multi_decision += 1
        if pred[i] != ground_truth[i]:
          TN+=1
          Detect+=1
          g_TN+=1
          g_Detect+=1
        else:
          FN+=1
          FN_result.append(["Rule 2","sample:",i+1,"Ground:",ground_truth[i],"Us:",top_camera,"Count:",top_camera_count,"Pred:",pred[i]])
          g_FN+=1
  
  
      all_result.insert(i,[i+1,top_camera,top_camera_count,pred[i]])
    # g_all_result.insert(int(M)-1,["Model:",M,"TP:",TP,"FP:",FP,"TN:",TN,"FN:",FN,"Wrong:",Wrong,"Detect:",Detect,"Fix:",Fix,"Model Fault:",model_error])
    g_all_result.insert(int(M)-1,["Model:",M,"Confirm(TP):",TP,"Fix:",Fix,"Can't Detect(FN):",FN,"Wrong(FP):",FP,"Detect(TN):",TN, "Model Fault:",model_error])
    print("TP:",TP)
    print("FP:",FP)
    print("TN:",TN)
    print("FN:",FN)
    print("Fix:",Fix)
    print("Wrong:",Wrong)
    print("Detect:",Detect)
    print("Sum:",TP+FP+TN+FN)
  
  
