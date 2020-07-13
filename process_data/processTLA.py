from ROOT import RDataFrame, TTree
import pandas as pd

def processTLA():
    #Load ROOT File and make it a dataframe
    path_to_data = '/afs/cern.ch/work/s/sarobert/autoencoders/inRootFiles/'
    folder = 'data18_13TeV.00364292.calibration_DataScouting_05_Jets.deriv.DAOD_TRIG6.r10657_p3592_p3754/'
    fname = 'DAOD_TRIG6.16825104._000001.pool.root.1'
    filePath = path_to_data + folder + fname
    treeName = 'CollectionTree'
    prefix ='HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn'
    df = RDataFrame(treeName, filePath)

    #Define Cuts
    leadpT = prefix + '.pt[0]'
    subpT = prefix + '.pt[1]'
    leadEta = prefix + '.eta[0]' #Using eta ~= y
    subEta = prefix + '.eta[1]'
    ystar = '1./2. * abs(' + leadEta + ' - ' + subEta + ')'
    cut1 = '(' + leadpT + ' > 185000) && (' + subpT + ' > 85000) && (' + ystar + ' < .3)'
    print ('Cut to be applied:')
    print (cut1)
    
    #Make Cuts
    print (str(df.Count().GetValue()) + ' events initially') 
    cut_df = df.Filter(cut1)
    print (str(cut_df.Count().GetValue()) + ' events passed the cuts')
    
    #cut_df.Snapshot('TLA_cuts.root')
    #Convert to numpy and save
    df_np = cut_df.AsNumpy()
   # df_pd = pd.DataFrame(cut_df)
#    df_pd.to_pickle('TLA_Cut.pkl')

processTLA()
