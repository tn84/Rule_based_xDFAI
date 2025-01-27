# Rule_based_xDFAI

## Important notes:
Set the required initial values based on your model.
For example: the number of classes, class names, paths, the number of layers, and layer names.

## To run the framework on your model, follow the steps in the following order:
1- Important features are extracted by Imf_Extraction.py.<br />
2- Traces and their associated indices are identified by functions in Trace Detection.py. <br />
3- Activation values (for both train and test samples) of the identified Traces are extracted by Trace_Value_Extraction. <br />
4- Euclidean distances, both normal and strict, are calculated in NED_SED.py. <br />
 5- Majority voting and model identification based on rules are performed by functions in Majority_Voting_Rules. <br />
6- Finally, the results of the proposed framework will be visible in xDFAI.py. <br />

The order of executing the modules is specified in  xDFAI.py.
