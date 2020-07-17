#Converts a DataScouting dataset to a numpy array to be fed into NN. Due to packaging issues, if running on lxplus, use python 2.7 as it has both ROOT and pandas installed centrally
import ROOT
from ROOT import RDataFrame
import pandas as pd

def processTLA():
    #Load ROOT File and make it a dataframe
    path_to_data = '/afs/cern.ch/work/s/sarobert/autoencoders/inRootFiles/data18_13TeV.00364292.calibration_DataScouting_05_Jets.deriv.DAOD_TRIG6.r10657_p3592_p3754/'
    fname1 = 'DAOD_TRIG6.16825104._0000'
    fname2 = '.pool.root.1'
    #filePath = path_to_data + folder + fname
    treeName = 'CollectionTree'
    prefix ='HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn'
    #files = ROOT.std.vector("string")(10)
#    for i in range(10):
#        if (i == 9):
#            files[i] = path_to_data + fname1 + str(i+1) + fname2
#        else:
#            files[i] = path_to_data + fname1 + '0' + str(i+1) + fname2
        
    files = path_to_data + fname1 + '16' + fname2
    print (files)
    df = RDataFrame(treeName, files)

    branchnames = [
    # 4-momentum
    'pt',
    'eta',
    'phi',
    'm',
    # Energy deposition in each calorimeter layer
    'EnergyPerSampling',
    # Area of jet,used for pile-up suppression (4-vector)
    'ActiveArea',
    'ActiveArea4vec_eta',
    'ActiveArea4vec_m',
    'ActiveArea4vec_phi',
    'ActiveArea4vec_pt',
#    'JetGhostArea',
    # Variables related to quality of jet
    'AverageLArQF',
#    'BchCorrCell',
    'NegativeE',
    'HECQuality',
    'LArQuality',
    # Shape and position, most energetic cluster
    'Width',
    'WidthPhi',
    'CentroidR',
    'DetectorEta',
    'LeadingClusterCenterLambda',
    'LeadingClusterPt',
    'LeadingClusterSecondLambda',
    'LeadingClusterSecondR',
    'N90Constituents',
    # Energy released in each calorimeter
    'EMFrac',
    'HECFrac',
    # Variables related to the time of arrival of a jet
    'Timing',
    'OotFracClusters10',
    'OotFracClusters5',
]

    #Define Cuts
    leadpT = prefix + '.pt[0]'
    subpT = prefix + '.pt[1]'
    leadEta = prefix + '.eta[0]' #Using eta ~= y
    subEta = prefix + '.eta[1]'
    ystar = '1./2. * abs(' + leadEta + ' - ' + subEta + ')'
    cut1 = '(' + leadpT + ' > 185000) && (' + subpT + ' > 85000) && (' + ystar + ' < .3)'
#    print ('Cut to be applied:')
#    print (cut1)
    
    #Make Cuts
    print (str(df.Count().GetValue()) + ' events initially') 
    cut_df = df.Filter(cut1)
    print (str(cut_df.Count().GetValue()) + ' events passed the cuts')
    for var in branchnames:
        cut_df = cut_df.Define(var, prefix + '.' + var)

    #Convert to numpy and save
    df_np = cut_df.AsNumpy(branchnames)
    df_pd = pd.DataFrame(df_np)
    print ('converted to numpy')
    print (df_pd.head())

    df_dict = {}
    for pp, branchname in enumerate(branchnames):
        if 'EnergyPerSampling' in branchname:
            pass
        else:
            #variable = branchname.split('.')[1]
            df_dict[branchname] = []
            jaggedX = df_pd[branchname]
            if pp == 0:
                print (branchname)
                print(jaggedX)
            for ii, arr in enumerate(jaggedX):
                for kk, val in enumerate(arr):
                    df_dict[branchname].append(val)
    if pp % 3 == 0:
        print((pp * 100) // len(branchname), '%')
    print('100%')
    print('Creating DataFrame...')
    partial_df = pd.DataFrame(data=df_dict)
    print('done.')

    del df_dict

    print(partial_df.head())
    partial_df.to_pickle('TLAJets_testing.pkl')

processTLA()
