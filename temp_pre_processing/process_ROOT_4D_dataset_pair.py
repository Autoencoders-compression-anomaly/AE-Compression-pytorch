"""
This pickles two matching datasets, first matching up the eventnumbers and then the jets, so that the data is in the same order.
"""

import uproot
import numpy as np

file = uproot.open("data/user.elmsheus.20339024.EXT1._000003.DAOD_PHYS.data.pool.root")

#Compressed variables have a c in them somewhere
filec = uproot.open("data/user.elmsheus.20338515.EXT1._000003.DAOD_PHYS.data.pool.root")

feventNumber = file["CollectionTree/EventInfoAux.eventNumber"].array()
fceventNumber = filec["CollectionTree/EventInfoAux.eventNumber"].array()

fm = file["CollectionTree/AntiKt4EMTopoJetsAux.m"].array()
fphi = file["CollectionTree/AntiKt4EMTopoJetsAux.phi"].array()
feta = file["CollectionTree/AntiKt4EMTopoJetsAux.eta"].array()
fpt = file["CollectionTree/AntiKt4EMTopoJetsAux.pt"].array()


fcm = filec["CollectionTree/AntiKt4EMTopoJetsAux.m"].array()
fcphi = filec["CollectionTree/AntiKt4EMTopoJetsAux.phi"].array()
fceta = filec["CollectionTree/AntiKt4EMTopoJetsAux.eta"].array()
fcpt = filec["CollectionTree/AntiKt4EMTopoJetsAux.pt"].array()

fevents = [[feventNumber[i], fm[i], fphi[i], feta[i], fpt[i]] for i in range(0,len(feventNumber))]

fcevents = [[fceventNumber[i], fcm[i], fcphi[i], fceta[i], fcpt[i]] for i in range(0,len(fceventNumber))]

fevents.sort()
fcevents.sort()

if len(fevents) != len(fcevents):
	raise Exception("Number of events not matching")

for i in range(0,len(fevents)):
	if fevents[i][0] != fcevents[i][0]:
		raise Exception("Events are not matching")

#Match jets by deltaR < 0.05

def proper_phi(phi):
	if phi < -np.pi:
		return phi+2*np.pi
	if phi > np.pi:
		return phi-2*np.pi
	return phi

def deltaR(eta1, eta2, phi1, phi2):
	deta = eta1-eta2
	dphi = proper_phi(phi1-phi2)
	return np.sqrt(deta**2+dphi**2)

#Iterate through every event and sort jets according to the uncompressed data

compressed_events = []

#Iterate through events
for ei in range(0,len(fevents)):

	if ei % 1000 == 0:
		print(str(int(ei/len(fevents)*100)) + "%")
	
	cm = []
	cphi = []
	ceta = []
	cpt = []
	
	#Iterate through jets in uncompressed event
	for ji in range(0,len(fevents[ei][1])):
		for jci in range(0,len(fcevents[ei][1])):
			dR = deltaR(fevents[ei][3][ji], fcevents[ei][3][jci], fevents[ei][2][ji], fcevents[ei][2][jci])

			if dR < 0.05:
				cm.append(fcevents[ei][1][jci])
				cphi.append(fcevents[ei][2][jci])
				ceta.append(fcevents[ei][3][jci])
				cpt.append(fcevents[ei][4][jci])

				#fcevents[ei][1] = np.delete(fcevents[ei][1],jci)
				#fcevents[ei][2] = np.delete(fcevents[ei][2],jci)
				#fcevents[ei][3] = np.delete(fcevents[ei][3],jci)
				#fcevents[ei][4] = np.delete(fcevents[ei][4],jci)

	if len(fevents[ei][1]) == len(cm):	
		compressed_events.append([fevents[ei][0], cm, cphi, ceta, cpt])
	else:
		print("Jets not matched in event: " + str(fevents[ei][0]))

