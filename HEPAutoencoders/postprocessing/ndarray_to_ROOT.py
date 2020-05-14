"""
This script saves a numpy ndarray (a detached torch tensor)
of single jet events (i.e. not jagged arrays)
back to a ROOT TTree, without ROOT or Athena.

TODO: Metadata?, compressiontypes
"""

import uproot
import numpy

#Specifies the 27D dataset. The available 'columns' can be read with ttree.keys()
prefix = 'HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn'
branches27D = [
    # 4-momentum
    (prefix + '.pt',numpy.float64),
    (prefix + '.eta',numpy.float64),
    (prefix + '.phi',numpy.float64),
    (prefix + '.m',numpy.float64),
    # Energy deposition in each calorimeter layer
    # prefix + '.EnergyPerSampling',
    # Area of jet,used for pile-up suppression (4-vector)
    (prefix + '.ActiveArea',numpy.int32),
    (prefix + '.ActiveArea4vec_eta',numpy.float64),
    (prefix + '.ActiveArea4vec_m',numpy.float64),
    (prefix + '.ActiveArea4vec_phi',numpy.float64),
    (prefix + '.ActiveArea4vec_pt',numpy.float64),
    # prefix + '.JetGhostArea',
    # Variables related to quality of jet
    (prefix + '.AverageLArQF',numpy.float64),
    # prefix + '.BchCorrCell',
    (prefix + '.NegativeE',numpy.float64),
    (prefix + '.HECQuality',numpy.float64),
    (prefix + '.LArQuality',numpy.float64),
    # Shape and position, most energetic cluster
    (prefix + '.Width',numpy.float64),
    (prefix + '.WidthPhi',numpy.float64),
    (prefix + '.CentroidR',numpy.float64),
    (prefix + '.DetectorEta',numpy.float64),
    (prefix + '.LeadingClusterCenterLambda',numpy.float64),
    (prefix + '.LeadingClusterPt',numpy.float64),
    (prefix + '.LeadingClusterSecondLambda',numpy.float64),
    (prefix + '.LeadingClusterSecondR',numpy.float64),
    (prefix + '.N90Constituents',numpy.int32),
    # Energy released in each calorimeter
    (prefix + '.EMFrac',numpy.float64),
    (prefix + '.HECFrac',numpy.float64),
    # Variables related to the time of arrival of a jet
    (prefix + '.Timing',numpy.float64),
    (prefix + '.OotFracClusters10',numpy.float64),
    (prefix + '.OotFracClusters5',numpy.float64),
]


def ndarray_to_DxAOD(filename, array, branches=branches27D, compression=uproot.ZLIB):
    
    f = uproot.recreate(filename)
    
    branchdict = dict(branches)
    print(branchdict)
    
    f["CollectionTree"] = uproot.newtree(branchdict)
    #for i,branch  in enumerate(branches):
    #    data = array[:,i]
    #    print(branch[0])
    f["CollectionTree"].extend(dict([(branch[0],array[:,i]) for (i,branch) in enumerate(branches)]))
    

