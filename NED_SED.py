# 7A 7B -2


layers = model_layers
Cmodels = class_name
shap_type = 'ch'  # none,ch
mode= 'Gradient'  # Gradient, Deep, Grad-CAM
prc=0.05
path="Results/All_Outputs/Model-"
M= "target class for the sample test"
test_path = f"Results/All_Outputs/Model-{M}/Test_Activation-{M}"
Method = 'Euclidean' # Euclidean , Static

with open(test_path+'/Ground_truth.pkl', 'rb') as file:
    ground_truth = pickle.load(file)

with open(test_path+'/Prediction.pkl', 'rb') as file:
    pred = pickle.load(file)

count_modelA = []
percent_modelA = []
count_modelB = []
percent_modelB = []

for m, model in enumerate(Cmodels):
  count_layerA = []
  percent_layerA = []
  count_layerB = []
  percent_layerB = []
  print('===================================')
  print('>>> Class:',model,'<<<')
  print('===================================')
  for l, layer in enumerate(layers):
    percent_sampleA = []
    count_sampleA = []
    percent_sampleB = []
    count_sampleB = []
    print('### layer:',layer,'###')
    N_top = load_result(path + model + '/N_top_Model_' + model + '_' + layer + '_' + mode + '_'+ shap_type +'_N_'+str(prc)+'.pickle')
    N = int(N_top*prc)
    shap_top_indices = load_result(path + model + '/shap_top_indices_Model_' + model + '_' + layer + '_' + mode + '_'+ shap_type +'_N_'+str(prc)+'.pickle')
    shap_top_values = load_result(path + model + '/shap_top_values_Model_' + model + '_' + layer + '_' + mode + '_'+ shap_type +'_N_'+str(prc)+'.pickle')
    shap_top_indices_flat = load_result(path + model + '/shap_top_indices_flat_Model_' + model + '_' + layer + '_' + mode + '_'+ shap_type +'_N_'+str(prc)+'.pickle')
    activation_top_values = load_result(path + model + '/activation_top_values_Model_' + model + '_' + layer + '_' + mode + '_'+ shap_type +'_N_'+str(prc)+'.pickle')
    test_top_values = test_top_value(layer,test_path+'/'+layer+'.npy')
    Std_top_valuesB = load_result(path + model + '/Std2_top_values_Model_' + model + '_' + layer + '_' + mode + '_'+ shap_type +'_N_'+str(prc)+'.pickle')
    print('-----------------------------------------------------------------------')
    for i in range(len(ground_truth)):
      countA = 0
      percentA=0
      countB = 0
      percentB=0
      weighted_countA = 0
      weighted_countB = 0
      weighted_indices = 0
      shap = np.transpose(shap_top_values, (1,0))
      for j, idx in enumerate(shap_top_indices):
        weighted_indices = weighted_indices + (shap_top_indices.index(idx)+1)
        similaritiesA = [math.dist([test_top_values[i][j]], [x]) for x in activation_top_values[j]]
        similaritiesB =[math.dist([test_top_values[i][j]], [np.median(activation_top_values[j])])]
        most_similar = min(similaritiesA)
        thresholdB = Std_top_valuesB[j]

        if most_similar <= thresholdB:
          countA = countA + 1
          weighted_countA = weighted_countA + (shap_top_indices.index(idx)+1)

        if similaritiesB <= thresholdB:
          countB = countB + 1
          weighted_countB = weighted_countB + (shap_top_indices.index(idx)+1)

      if weighted_countA >= int(weighted_indices/2):
        count_sampleA.insert(i,[countA,len(shap_top_indices),weighted_countA, weighted_indices])
      else:
        count_sampleA.insert(i,[0,len(shap_top_indices),0, weighted_indices])

      if weighted_countA >= int(weighted_indices/2):
        percentA=round((countA/len(shap_top_indices))*100,2)
        percent_sampleA.insert(i,[percentA,model])  # Keep percent and model of all samples for current layer
      else:
        percent_sampleA.insert(i,[0,model])

      if weighted_countB >= int(weighted_indices/2):
        count_sampleB.insert(i,[countB,len(shap_top_indices),weighted_countB, weighted_indices])
      else:
        count_sampleB.insert(i,[0,len(shap_top_indices),0, weighted_indices])

      if weighted_countB >= int(weighted_indices/2):
        percentB=round((countB/len(shap_top_indices))*100,2)
        percent_sampleB.insert(i,[percentB,model])  # Keep percent and model of all samples for current layer
      else:
        percent_sampleB.insert(i,[0,model])

    count_layerA.insert(l,count_sampleA)
    percent_layerA.insert(l,percent_sampleA) # Keep percent and model of all layers for current model

    count_layerB.insert(l,count_sampleB)
    percent_layerB.insert(l,percent_sampleB) # Keep percent and model of all layers for current model

  count_modelA.insert(m,count_layerA)
  percent_modelA.insert(m,percent_layerA) # Keep percent and model of all models

  count_modelB.insert(m,count_layerB)
  percent_modelB.insert(m,percent_layerB) # Keep percent and model of all models

# save results
save_result(count_modelA,path+f"{M}/Results/{Method}/count_model-7A-less.pickle")
save_result(percent_modelA,path+f"{M}/Results/{Method}/percent_model-7A-less.pickle")
save_result(count_modelB,path+f"{M}/Results/{Method}/count_model-7B-less.pickle")
save_result(percent_modelB,path+f"{M}/Results/{Method}/percent_model-7B-less.pickle")
