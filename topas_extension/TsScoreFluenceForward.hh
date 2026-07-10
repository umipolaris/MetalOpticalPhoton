// TsScoreFluenceForward — Fluence with forward-direction filter
//
// Standard TsScoreFluence accumulates step length for every photon regardless
// of momentum direction. When measuring a collimated beam through reflective/
// refractive media, backward Fresnel-reflected photons also accumulate into
// the same parallel-world voxels (정상 물리 but obscures forward beam shape).
//
// This subclass filters: ProcessHits only accepts steps with momentum
// direction along configured axis (default +Z). Other steps skipped.
//
// Usage in TOPAS config:
//   s:Sc/XYscore/Quantity = "FluenceForward"
//   # optional:
//   sv:Sc/XYscore/ForwardAxis = 3 0.0 0.0 1.0   # default
//   d:Sc/XYscore/MinCosTheta  = 0.0             # default = forward半hemisphere

#ifndef TsScoreFluenceForward_hh
#define TsScoreFluenceForward_hh

#include "TsVBinnedScorer.hh"
#include "G4ThreeVector.hh"

class TsScoreFluenceForward : public TsVBinnedScorer
{
public:
    TsScoreFluenceForward(TsParameterManager* pM, TsMaterialManager* mM,
                          TsGeometryManager* gM, TsScoringManager* scM,
                          TsExtensionManager* eM,
                          G4String scorerName, G4String quantity,
                          G4String outFileName, G4bool isSubScorer);
    virtual ~TsScoreFluenceForward();

    G4bool ProcessHits(G4Step*, G4TouchableHistory*);

private:
    G4ThreeVector fForwardAxis;
    G4double      fMinCosTheta;
};
#endif
