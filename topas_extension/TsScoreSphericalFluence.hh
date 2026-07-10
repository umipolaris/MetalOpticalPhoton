/**
 * TsScoreSphericalFluence — sphere component 의 R/Phi/Theta voxel grid 정확
 * 분배 새 scorer (2026-05-16).
 *
 * 배경 — TOPAS 기본 TsScoreFluence + TsSphere 의 G4 navigator 가 nested
 * concentric R divisions 시 step 길이 분할 + GetIndex copyNo 모두 wrong
 * (TsCylinder Rho/Phi divisions 의 known TOPAS 3.6.0 bug 와 same family).
 * 이 scorer 는 navigator copyNo 무시 + step ray 받아 R/Phi/Theta sample-based
 * voxel path 직접 분배. GPU GPUOpticalPhotonFluence 와 같은 logic.
 *
 * TOPAS 파라미터:
 *   s:Sc/MyScorer/Quantity = "SphericalFluence"
 *   s:Sc/MyScorer/Component = "SomeSphereOrHemisphere"
 *   i:Sc/MyScorer/RBins     = N
 *   i:Sc/MyScorer/PhiBins   = N
 *   i:Sc/MyScorer/ThetaBins = N
 *
 * Component 는 TsSphere 여야 함 (G4Sphere envelope).
 */

#ifndef TS_SCORE_SPHERICAL_FLUENCE_HH
#define TS_SCORE_SPHERICAL_FLUENCE_HH

#include "TsVBinnedScorer.hh"

class TsScoreSphericalFluence : public TsVBinnedScorer
{
public:
    TsScoreSphericalFluence(TsParameterManager* pM, TsMaterialManager* mM,
                            TsGeometryManager* gM, TsScoringManager* scM,
                            TsExtensionManager* eM,
                            G4String scorerName, G4String quantity,
                            G4String outFileName, G4bool isSubScorer);

    ~TsScoreSphericalFluence() override;

    G4bool ProcessHits(G4Step*, G4TouchableHistory*) override;

private:
    void CacheGeometry();

    G4double fCenterX, fCenterY, fCenterZ;  // sphere center (world coord)
    G4double fRMin, fRMax;
    G4double fSPhi, fDPhi;
    G4double fSTheta, fDTheta;
    G4int    fNRBins, fNPhiBins, fNThetaBins;
    G4bool   fGeomCached;
};

#endif
