// Scorer for FluenceForward
#include "TsScoreFluenceForward.hh"

TsScoreFluenceForward::TsScoreFluenceForward(TsParameterManager* pM,
        TsMaterialManager* mM, TsGeometryManager* gM, TsScoringManager* scM,
        TsExtensionManager* eM, G4String scorerName, G4String quantity,
        G4String outFileName, G4bool isSubScorer)
: TsVBinnedScorer(pM, mM, gM, scM, eM, scorerName, quantity,
                  outFileName, isSubScorer),
  fForwardAxis(0., 0., 1.), fMinCosTheta(0.0)
{
    SetUnit("/mm2");

    // Optional: forward axis (default +Z)
    G4String axisName = GetFullParmName("ForwardAxis");
    if (fPm->ParameterExists(axisName)) {
        G4int n = fPm->GetVectorLength(axisName);
        if (n == 3) {
            G4double* v = fPm->GetUnitlessVector(axisName);
            fForwardAxis = G4ThreeVector(v[0], v[1], v[2]).unit();
        }
    }
    // Optional: min cos(theta) threshold (default 0 → forward hemisphere)
    G4String cosName = GetFullParmName("MinCosTheta");
    if (fPm->ParameterExists(cosName)) {
        fMinCosTheta = fPm->GetUnitlessParameter(cosName);
    }
}

TsScoreFluenceForward::~TsScoreFluenceForward() {;}

G4bool TsScoreFluenceForward::ProcessHits(G4Step* aStep, G4TouchableHistory*)
{
    if (!fIsActive) {
        fSkippedWhileInactive++;
        return false;
    }

    // Forward-direction filter: pre-step momentum dot forward-axis
    const G4ThreeVector& dir = aStep->GetPreStepPoint()->GetMomentumDirection();
    if (dir.dot(fForwardAxis) <= fMinCosTheta) return false;

    G4double quantity = aStep->GetStepLength();
    if (quantity > 0.) {
        ResolveSolid(aStep);
        quantity /= GetCubicVolume(aStep);
        quantity *= aStep->GetPreStepPoint()->GetWeight();
        AccumulateHit(aStep, quantity);
        return true;
    }
    return false;
}
