import sys, getopt
import httpagentparser
import numpy as np
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from time import time
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

def main(argv):

    training_file = ''
    test_file = ''
    prediction_file = ''
    
    try:
        opts, args = getopt.getopt(argv, "h", ["help", "training=", "test=", "prediction="])
    except getopt.GetoptError:
        print('train.py --training <training_file> --test <test_file> --prediction <prediction_file>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ["-h", "--help"]:
         print('train.py --training <training_file> --test <test_file> --prediction <prediction_file>')
         sys.exit()
        elif opt == "--training":
            training_file = arg
        elif opt == "--test":
            test_file = arg
        elif opt == "--prediction":
            prediction_file = arg  
    if((not training_file) or (not test_file) or (not prediction_file)): 
        print("Missing input parameters")
        sys.exit()
    else:
        print("Training data file is \"" + training_file + "\"")
        print("Test data file is \"" + test_file + "\"")
        print("Prediction file is \"" + prediction_file + "\"")
        print("")
    
    
    ftrain = open(training_file,'r')
    UA_train, Family_train, Version_train, os_train, pf_train, dist_train, bro_train = parseUA(ftrain)
    ftrain.close()
    
    ftest = open(test_file,'r')
    UA_test, Family_test, Version_test, os_test, pf_test, dist_test, bro_test = parseUA(ftest)
    ftest.close()
    
    
    #Build DictVectorizer to transform categorical features to vector
    vec = DictVectorizer()
    
    #Use training data to build DictVectorizer and transform for both training data and test data
    osFit = vec.fit(os_train)
    osFeatureNames = osFit.get_feature_names()
    
    pfFit = vec.fit(pf_train)
    pfFeatureNames = pfFit.get_feature_names()
    
    distFit = vec.fit(dist_train)
    distFeatureNames = distFit.get_feature_names()
    
    broFit = vec.fit(bro_train)
    broFeatureNames = broFit.get_feature_names()
    
    featureNames = np.concatenate((osFeatureNames, pfFeatureNames, distFeatureNames, broFeatureNames))
    
    #Transform training data and test data
    osArray_train = osFit.transform(os_train).toarray()
    pfArray_train = pfFit.transform(pf_train).toarray()
    distArray_train = distFit.transform(dist_train).toarray()
    broArray_train = broFit.transform(bro_train).toarray()
    
    osArray_test = osFit.transform(os_test).toarray()
    pfArray_test = pfFit.transform(pf_test).toarray()
    distArray_test = distFit.transform(dist_test).toarray()
    broArray_test = broFit.transform(bro_test).toarray()

    
    #print(osArray_train.shape, pfArray_train.shape, distArray_train.shape, broArray_train.shape)
    #print(np.isnan(np.sum(osArray_train)), np.isnan(np.sum(pfArray_train)), np.isnan(np.sum(distArray_train)), np.isnan(np.sum(broArray_train)))

    #Compose feature matrix
    print("Building feature matrix:")
    X_train = np.concatenate((osArray_train, pfArray_train, distArray_train, broArray_train), axis=1)
    X_test = np.concatenate((osArray_test, pfArray_test, distArray_test, broArray_test), axis=1)
    

    #Build lableEncoding for user agent family by using training data  
    le = preprocessing.LabelEncoder()
    le.fit(Family_train)
    #print(le.classes_)
    
    #Transform 
    y_train = np.asarray(le.transform(Family_train))
    y_test = np.asarray(le.transform(Family_test))
    
    #Build user agent version label
    z_train = np.asarray(Version_train)
    z_test = np.asarray(Version_test)

#    print('Saving training data to file:')
#    np.savetxt('feature.txt', X)
#    np.savetxt('family.txt', np.asarray(le.transform(Family_train)))
#    np.savetxt('version.txt', np.asarray(totalVersion))
    
    
    #For cross validation, comment it for using real test data
    #print("Splitting data for cross validation:")
    #X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(X_train, y_train, z_train, test_size=0.30, random_state=42)
    
    #X_train, X_test, y_train, y_test, z_train, z_test, UA_train, UA_test, Family_train, Family_test, Version_train, Version_test = \
    #train_test_split(X_train, y_train, z_train, UA_train, Family_train, Version_train, test_size=0.30, random_state=42)
    
    
    
    max_depth_family = None
    n_estimators_family = 10
    
    """
    #------- use a full grid over all parameters ------------
    print("Using grid search to find optimal parameters")
    param_grid = {
              "min_samples_split": [10, 20, 30],
              "warm_start": [True, False],
              "criterion": ["gini", "entropy"],
              "bootstrap": [True, False]}
              
              
    param_grid = {"max_depth": [30, None],
              "min_samples_split": [2, 10],
              "min_samples_leaf": [1, 5],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"],
              "class_weight":["balanced", "balanced_subsample", None],
              "warm_start": [True, False]}         
    
    clf = RandomForestClassifier(n_estimators = n_estimators_family, n_jobs=-1)
    # run grid search
    grid_search = GridSearchCV(clf, param_grid=param_grid)
    
    start = time()
    grid_search.fit(X_train, y_train)                               
    print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
    report(grid_search.cv_results_)
    """
    
    
    
    print("Stating training for user agent family by random forest:")
    
    rf_model_family = RandomForestClassifier(n_estimators = n_estimators_family, max_depth = max_depth_family, n_jobs=-1, random_state = 2, criterion="entropy", min_samples_split=10)
    rf_model_family.fit(X_train, y_train)
    
 
    
    #Prediction
    y_pred_family_train = rf_model_family.predict(X_train)
    score_family_train = accuracy_score(y_train, y_pred_family_train)
    print('-----------------------------------------------------------------------------')
    print('Accuracy for user agent family in training data: {0}'.format(score_family_train))
    print('-----------------------------------------------------------------------------')
    print("")
    
    
    print("Stating training for user agent version by random forest:")
    max_depth_version = None
    n_estimators_version = 10
    
    rf_model_version = RandomForestClassifier(n_estimators = n_estimators_version, max_depth = max_depth_version, n_jobs=-1, random_state = 41, criterion="entropy", min_samples_split=10)
    rf_model_version.fit(X_train, z_train)    
    
    #Prediction
    z_pred_version_train = rf_model_version.predict(X_train)
    score_version_train = accuracy_score(z_train, z_pred_version_train)
    print('-----------------------------------------------------------------------------')
    print('Accuracy for user agent version in training data: {0}'.format(score_version_train))
    print('-----------------------------------------------------------------------------')
    
    
    #Prediction for test data
    print("Make prediction for test data:")
    
    y_pred_family_test = rf_model_family.predict(X_test)
    score_family_test = accuracy_score(y_test, y_pred_family_test)
    print('-----------------------------------------------------------------------------')
    print('Accuracy for user agent family in test data: {0}'.format(score_family_test))
    print('-----------------------------------------------------------------------------')
    print("")
    
    z_pred_version_test = rf_model_version.predict(X_test)
    score_version_test = accuracy_score(z_test, z_pred_version_test)
    print('-----------------------------------------------------------------------------')
    print('Accuracy for user agent version in test data: {0}'.format(score_version_test))
    print('-----------------------------------------------------------------------------')
    
    
    #Output file prediction file 
    fw = open(prediction_file,'w')
    
    family_pred_label = le.inverse_transform(y_pred_family_test).tolist()
    version_pred_label = z_pred_version_test.astype(int).tolist()
    
    print("Writing to output file:")
    for i in range(len(UA_test)):
        #print(UA_test[i], Family_test[i], str(int(Version_test[i])), family_pred_label[i], str(version_pred_label[i]))
        fw.write(UA_test[i] + '\t' + Family_test[i] + '\t' + str(int(Version_test[i])) + '\t' + family_pred_label[i] + '\t' + str(version_pred_label[i]) + '\n')
    fw.close()
    
    print("-------------------------------- Finished ------------------------------------")
    
    
    
    


#Function fo parse user agent string    
def parseUA(inputFile):
    
    print("Parsing user agent string:")
    
    #Create global list to store each line
    #Prepare data taking version as features
    osDict = []
    pfDict = []
    distDict = []
    broDict = []
    
    totalUA = []
    totalFamily = []
    totalVersion = []
    
    ualist = []
    
    for line in inputFile:
        line = line.rstrip('\n')
        line = line.rstrip()
        line = line.lstrip()
        lineSplitted = line.split('\t')
        if(len(lineSplitted) == 3 and lineSplitted[2].isdigit()):
            uaString, uaFamily, uaVersion = lineSplitted
            totalUA.append(uaString)
            totalFamily.append(uaFamily)
            totalVersion.append(float(uaVersion))
        else:   continue

        if uaString != '':
            #Use httpagentparser package to parse user agent
            uaDict = httpagentparser.detect(uaString)
            ualist.append(uaDict)
            #print(uaDict)
            
            
            
            if 'os' in uaDict and uaDict['os']['name'] != None:
                osDictOrig = uaDict['os']
                if 'version' in osDictOrig:
                    if osDictOrig['version'] != None:
                        versionNum = osDictOrig['version'].split('.')
                        if(versionNum[0] != ''):
                            osDictOrig['version'] = versionNum[0]
                    else: del osDictOrig['version']
                osDict.append(osDictOrig)
            else: 
                osDict.append({})

            
            
            if 'platform' in uaDict and uaDict['platform']['name'] != None:
                pfDictOrig = uaDict['platform']
                if 'version' in pfDictOrig:
                    if pfDictOrig['version'] != None:
                        versionNum = pfDictOrig['version'].split('.')
                        if(versionNum[0] != ''):
                            pfDictOrig['version'] = versionNum[0]
                    else: del pfDictOrig['version']
                pfDict.append(pfDictOrig)
            else: 
                pfDict.append({})


            
            if 'dist' in uaDict and uaDict['dist']['name'] != None:
                distDictOrig = uaDict['dist']
                if 'version' in distDictOrig:
                    if distDictOrig['version'] != None:
                        versionNum = distDictOrig['version'].split('.')
                        if(versionNum[0] != ''):
                            distDictOrig['version'] = versionNum[0]
                    else: del distDictOrig['version']
                distDict.append(distDictOrig)
            else: 
                distDict.append({})

            
            
            if 'browser' in uaDict and uaDict['browser']['name'] != None:
                broDictOrig = uaDict['browser']
                if 'version' in broDictOrig:
                    if broDictOrig['version'] != None:
                        versionNum = broDictOrig['version'].split('.')
                        if(versionNum[0] != ''):
                            broDictOrig['version'] = versionNum[0]
                    else: del broDictOrig['version']
                broDict.append(broDictOrig)
            else: 
                broDict.append({})
    
    return totalUA, totalFamily, totalVersion, osDict, pfDict, distDict, broDict
    

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
    
    

if __name__=="__main__":
    main(sys.argv[1:])
