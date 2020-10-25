#!/bin/python
#Assorted utils to process ATLAS jet data

import numpy as np

def unit_convert_jets(leading, subleading):
    leading_orig = leading.copy()
    leading['pt'] = leading['pt'] / 1000.  # Convert to GeV
    subleading['pt'] = subleading['pt'] / 1000.  # Convert to GeV
    leading_orig['pt'] = leading_orig['pt'] / 1000.  # Convert to GeV
    leading['m'] = leading['m'] / 1000.  # Convert to GeV
    subleading['m'] = subleading['m'] / 1000.  # Convert to GeV
    leading_orig['m'] = leading_orig['m'] / 1000.  # Convert to GeV
    leading['LeadingClusterPt'] = leading['LeadingClusterPt'] / 1000.  # Convert to GeV
    subleading['LeadingClusterPt'] = subleading['LeadingClusterPt'] / 1000.  # Convert to GeV
    leading_orig['LeadingClusterPt'] = leading_orig['LeadingClusterPt'] / 1000.  # Convert to GeV
    leading['LeadingClusterSecondR'] = leading['LeadingClusterSecondR'] / 1000.  # Convert to GeV
    subleading['LeadingClusterSecondR'] = subleading['LeadingClusterSecondR'] / 1000.  # Convert to GeV
    leading_orig['LeadingClusterSecondR'] = leading_orig['LeadingClusterSecondR'] / 1000.  # Convert to GeV
    leading['LeadingClusterSecondLambda'] = leading['LeadingClusterSecondLambda'] / 1000.  # Convert to GeV
    subleading['LeadingClusterSecondLambda'] = subleading['LeadingClusterSecondLambda'] / 1000.  # Convert to GeV
    leading_orig['LeadingClusterSecondLambda'] = leading_orig['LeadingClusterSecondLambda'] / 1000.  # Convert to GeV
    leading['NegativeE'] = leading['NegativeE'] / 1000.  # Convert to GeV
    subleading['NegativeE'] = subleading['NegativeE'] / 1000.  # Convert to GeV
    leading_orig['NegativeE'] = leading_orig['NegativeE'] / 1000.  # Convert to GeV

def filter_unitconvert_jets_4D(train):
    train['pt'] = train['pt'] / 1000.  # Convert to GeV
    train['m'] = train['m'] / 1000.  # Convert to GeV
    # Remove all jets with mass <= 0 (arbitrarily close-to-zero)
    train = train[(np.abs(train['m']) < 0.000000001)]
    return train

def filter_jets(train):
    train['pt'] = train['pt'] / 1000.  # Convert to GeV
    train['m'] = train['m'] / 1000.  # Convert to GeV
    train['LeadingClusterPt'] = train['LeadingClusterPt'] / 1000.  # Convert to GeV
    train['LeadingClusterSecondR'] = train['LeadingClusterSecondR'] / 1000.  # Convert to GeV
    train['LeadingClusterSecondLambda'] = train['LeadingClusterSecondLambda'] / 1000.  # Convert to GeV
    train['NegativeE'] = train['NegativeE'] / 1000.  # Convert to GeV

    if 'JetGhostArea' in train.keys():
        train.pop('JetGhostArea')
    if 'BchCorrCell' in train.keys():
        train.pop('BchCorrCell')

    # Remove all jets with EMFrac outside (-2, 2)
    train = train[(np.abs(train['EMFrac']) < 5)]
    train = train[np.invert((np.abs(train['EMFrac']) < 0.05) & (np.abs(train['eta']) >= 2))]
    train = train[np.invert((train['AverageLArQF'] > .8) & (train['EMFrac'] > .95) & (train['LArQuality'] > .8) & (np.abs(train['eta']) < 2.8))]
    train = train[np.abs(train['NegativeE']) < 60 * 5]

    # Filter out extreme jets
    train = train[np.invert((train['AverageLArQF'] > .8) & (np.abs(train['HECQuality']) > 0.5) & (np.abs(train['HECFrac']) > 0.5))]
    train = train[train['OotFracClusters10'] > -0.1]
    train = train[train['OotFracClusters5'] > -0.1]
    if 'Width' in train.keys():
        train = train[np.abs(train['Width']) < 5]
        train = train[np.invert(train['Width'] == -1)]
    if 'WidthPhi' in train.keys():
        train = train[np.abs(train['WidthPhi']) < 5]
    train = train[np.abs(train['Timing']) < 125]
    train = train[train['LArQuality'] < 4]
    train = train[np.abs(train['HECQuality']) < 2.5]
    # train = train[train['m'] > 1e-3]

    return train

# Custom normalization for AOD data
eta_div = 5
emfrac_div = 1.6
negE_div = 1.6
phi_div = 3
m_div = 1.8
width_div = .6
N90_div = 20
timing_div = 40
hecq_div = 1
centerlambda_div = 2
secondlambda_div = 1
secondR_div = .6
larqf_div = 2.5
pt_div = 1.2
centroidR_div = 0.8
area4vecm_div = 0.18
area4vecpt_div = 0.7
area4vec_div = 0.8
Oot_div = 0.3
larq_div = 0.6

log_add = 100
log_sub = 2
m_add = 1
centroidR_sub = 3
pt_sub = 1.3
area4vecm_sub = 0.15

#4D (some code duplication, but can clean up later)
def custom_normalization_4D(train, test):
    train_cp = train.copy()
    test_cp = test.copy()

    for data in [train_cp, test_cp]:
        data['eta'] = data['eta'] / eta_div
        data['phi'] = data['phi'] / phi_div
        data['m'] = np.log10(data['m'] + m_add) / m_div
        data['pt'] = (np.log10(data['pt']) - pt_sub) / pt_div

    return train_cp, test_cp

#27D
def custom_normalization(train, test):
    train_cp = train.copy()
    test_cp = test.copy()

    for data in [train_cp, test_cp]:
        data['DetectorEta'] = data['DetectorEta'] / eta_div
        data['ActiveArea4vec_eta'] = data['ActiveArea4vec_eta'] / eta_div
        data['EMFrac'] = data['EMFrac'] / emfrac_div
        data['NegativeE'] = np.log10(-data['NegativeE'] + 1) / negE_div
        data['eta'] = data['eta'] / eta_div
        data['phi'] = data['phi'] / phi_div
        data['ActiveArea4vec_phi'] = data['ActiveArea4vec_phi'] / phi_div
        if 'Width' in data.keys():
            data['Width'] = data['Width'] / width_div
        else:
            print('Wdith not found when normalizing')
        if 'WidthPhi' in data.keys():
            data['WidthPhi'] = data['WidthPhi'] / width_div
        else:
            print('WdithPhi not found when normalizing')
        data['N90Constituents'] = data['N90Constituents'] / N90_div
        data['Timing'] = data['Timing'] / timing_div
        data['HECQuality'] = data['HECQuality'] / hecq_div
        data['ActiveArea'] = data['ActiveArea'] / area4vec_div
        data['ActiveArea4vec_m'] = data['ActiveArea4vec_m'] / area4vecm_div - area4vecm_sub
        data['ActiveArea4vec_pt'] = data['ActiveArea4vec_pt'] / area4vecpt_div
        data['LArQuality'] = data['LArQuality'] / larq_div

        data['m'] = np.log10(data['m'] + m_add) / m_div
        data['LeadingClusterCenterLambda'] = (np.log10(data['LeadingClusterCenterLambda'] + log_add) - log_sub) / centerlambda_div
        data['LeadingClusterSecondLambda'] = (np.log10(data['LeadingClusterSecondLambda'] + log_add) - log_sub) / secondlambda_div
        data['LeadingClusterSecondR'] = (np.log10(data['LeadingClusterSecondR'] + log_add) - log_sub) / secondR_div
        data['AverageLArQF'] = (np.log10(data['AverageLArQF'] + log_add) - log_sub) / larqf_div
        data['pt'] = (np.log10(data['pt']) - pt_sub) / pt_div
        data['LeadingClusterPt'] = np.log10(data['LeadingClusterPt']) / pt_div
        data['CentroidR'] = (np.log10(data['CentroidR']) - centroidR_sub) / centroidR_div
        data['OotFracClusters10'] = np.log10(data['OotFracClusters10'] + 1) / Oot_div
        data['OotFracClusters5'] = np.log10(data['OotFracClusters5'] + 1) / Oot_div

    return train_cp, test_cp

def custom_unnormalize(normalized_data):
    data = normalized_data.copy()
    data['DetectorEta'] = data['DetectorEta'] * eta_div
    data['ActiveArea4vec_eta'] = data['ActiveArea4vec_eta'] * eta_div
    data['EMFrac'] = data['EMFrac'] * emfrac_div
    data['eta'] = data['eta'] * eta_div
    data['phi'] = data['phi'] * phi_div
    data['ActiveArea4vec_phi'] = data['ActiveArea4vec_phi'] * phi_div
    if 'Width' in data.keys():
        data['Width'] = data['Width'] * width_div
    else:
        print('Width not found when unnormalizing')
    if 'WidthPhi' in data.keys():
        data['WidthPhi'] = data['WidthPhi'] * width_div
    else:
        print('WidthPhi not found when unnormalizing')
    data['N90Constituents'] = data['N90Constituents'] * N90_div
    data['Timing'] = data['Timing'] * timing_div
    data['HECQuality'] = data['HECQuality'] * hecq_div
    data['ActiveArea'] = data['ActiveArea'] * area4vec_div
    data['ActiveArea4vec_m'] = (data['ActiveArea4vec_m'] + area4vecm_sub) * area4vecm_div
    data['ActiveArea4vec_pt'] = data['ActiveArea4vec_pt'] * area4vecpt_div
    data['LArQuality'] = data['LArQuality'] * larq_div

    data['NegativeE'] = 1 - np.power(10, negE_div * data['NegativeE'])
    data['m'] = np.power(10, m_div * data['m']) - m_add
    data['LeadingClusterCenterLambda'] = np.power(10, centerlambda_div * data['LeadingClusterCenterLambda'] + log_sub) - log_add
    data['LeadingClusterSecondLambda'] = np.power(10, secondlambda_div * data['LeadingClusterSecondLambda'] + log_sub) - log_add
    data['LeadingClusterSecondR'] = np.power(10, secondR_div * data['LeadingClusterSecondR'] + log_sub) - log_add
    data['AverageLArQF'] = np.power(10, larqf_div * data['AverageLArQF'] + log_sub) - log_add
    data['pt'] = np.power(10, pt_div * data['pt'] + pt_sub)
    data['LeadingClusterPt'] = np.power(10, pt_div * data['LeadingClusterPt'])
    data['CentroidR'] = np.power(10, centroidR_div * data['CentroidR'] + centroidR_sub)
    data['OotFracClusters10'] = np.power(10, Oot_div * data['OotFracClusters10']) - 1
    data['OotFracClusters5'] = np.power(10, Oot_div * data['OotFracClusters5']) - 1

    return data
