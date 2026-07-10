// Scorer for SphericalFluence
/**
 * TsScoreSphericalFluence — sphere R/Phi/Theta voxel grid 정확 분배.
 * GPU 의 DDA spherical voxel sample-based scoring 의 CPU equivalent.
 */

#include "TsScoreSphericalFluence.hh"

#include "TsParameterManager.hh"
#include "TsVGeometryComponent.hh"
#include "G4Step.hh"
#include "G4Sphere.hh"
#include "G4LogicalVolume.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"

#include <algorithm>
#include <cmath>
#include <iostream>

TsScoreSphericalFluence::TsScoreSphericalFluence(
    TsParameterManager* pM, TsMaterialManager* mM,
    TsGeometryManager* gM, TsScoringManager* scM,
    TsExtensionManager* eM,
    G4String scorerName, G4String quantity,
    G4String outFileName, G4bool isSubScorer)
: TsVBinnedScorer(pM, mM, gM, scM, eM, scorerName, quantity, outFileName, isSubScorer)
, fCenterX(0), fCenterY(0), fCenterZ(0)
, fRMin(0), fRMax(0), fSPhi(0), fDPhi(0), fSTheta(0), fDTheta(0)
, fNRBins(1), fNPhiBins(1), fNThetaBins(1)
, fGeomCached(false)
{
    SetUnit("/mm2");
}

TsScoreSphericalFluence::~TsScoreSphericalFluence() {}

void TsScoreSphericalFluence::CacheGeometry()
{
    if (fGeomCached) return;

    fNRBins     = fComponent->GetDivisionCount(0);
    fNPhiBins   = fComponent->GetDivisionCount(1);
    fNThetaBins = fComponent->GetDivisionCount(2);

    // Component envelope sphere 에서 직접 R/Phi/Theta 받음.
    G4LogicalVolume* lv = fComponent->GetEnvelopeLogicalVolume();
    G4VSolid* solid = lv ? lv->GetSolid() : nullptr;
    G4Sphere* envSph = dynamic_cast<G4Sphere*>(solid);
    if (!envSph) {
        G4cerr << "[TsScoreSphericalFluence] FATAL: Component '" << fComponentName
               << "' is not a TsSphere (envelope solid not G4Sphere)" << G4endl;
        fPm->AbortSession(1);
        return;
    }
    fRMin   = envSph->GetInnerRadius();
    fRMax   = envSph->GetOuterRadius();
    fSPhi   = envSph->GetStartPhiAngle();
    fDPhi   = envSph->GetDeltaPhiAngle();
    fSTheta = envSph->GetStartThetaAngle();
    fDTheta = envSph->GetDeltaThetaAngle();

    // sphere center (world coord) — TOPAS 파라미터에서 TransX/Y/Z 누적 (parent chain).
    G4String compName = fComponentName;
    size_t lastUnderscore = compName.rfind('_');
    if (lastUnderscore != std::string::npos) {
        G4String suffix = compName.substr(lastUnderscore + 1);
        if (suffix.find('x') != std::string::npos) {
            G4String originalName = compName.substr(0, lastUnderscore);
            if (fPm->ParameterExists("Ge/" + originalName + "/Parent"))
                compName = originalName;
        }
    }
    G4double tx = 0, ty = 0, tz = 0;
    while (true) {
        G4String prefix = "Ge/" + compName + "/";
        if (fPm->ParameterExists(prefix + "TransX"))
            tx += fPm->GetDoubleParameter(prefix + "TransX", "Length");
        if (fPm->ParameterExists(prefix + "TransY"))
            ty += fPm->GetDoubleParameter(prefix + "TransY", "Length");
        if (fPm->ParameterExists(prefix + "TransZ"))
            tz += fPm->GetDoubleParameter(prefix + "TransZ", "Length");
        if (!fPm->ParameterExists(prefix + "Parent")) break;
        compName = fPm->GetStringParameter(prefix + "Parent");
        if (compName == "World" || compName == "world") break;
    }
    fCenterX = tx;
    fCenterY = ty;
    fCenterZ = tz;

    fGeomCached = true;

    std::cout << "[TsScoreSphericalFluence] '" << fComponentName
              << "' geom cached: center=(" << fCenterX/mm << "," << fCenterY/mm
              << "," << fCenterZ/mm << ")mm  RMin=" << fRMin/mm << " RMax=" << fRMax/mm
              << "mm  SPhi=" << fSPhi << " DPhi=" << fDPhi
              << " STheta=" << fSTheta << " DTheta=" << fDTheta << "rad"
              << "  bins=(R=" << fNRBins << ", Phi=" << fNPhiBins
              << ", Theta=" << fNThetaBins << ")" << std::endl;
}

G4bool TsScoreSphericalFluence::ProcessHits(G4Step* aStep, G4TouchableHistory*)
{
    if (!fIsActive) { fSkippedWhileInactive++; return false; }

    G4double stepLen = aStep->GetStepLength();
    if (stepLen <= 0) return false;

    if (!fGeomCached) CacheGeometry();
    if (!fGeomCached) return false;

    G4ThreeVector preP  = aStep->GetPreStepPoint()->GetPosition();
    G4ThreeVector postP = aStep->GetPostStepPoint()->GetPosition();
    G4double weight = aStep->GetPreStepPoint()->GetWeight();

    G4ThreeVector ctr(fCenterX, fCenterY, fCenterZ);
    G4ThreeVector localPre  = preP  - ctr;
    G4ThreeVector localPost = postP - ctr;

    G4double binR     = (fNRBins > 0) ? (fRMax - fRMin) / fNRBins : 1.0;
    G4double binPhi   = (fNPhiBins > 0) ? fDPhi / fNPhiBins : 1.0;
    G4double binTheta = (fNThetaBins > 0) ? fDTheta / fNThetaBins : 1.0;

    // sample-based: step 의 N sample → 각 sample 의 (r, phi, theta) bin → contrib 분배.
    const int N = 256;
    G4double dseg = stepLen / N;
    G4bool anyHit = false;

    G4int  curBin   = -1;
    G4double curAccum = 0;

    for (int s = 0; s < N; s++) {
        G4double t = (s + 0.5) / G4double(N);
        G4ThreeVector p = localPre + (localPost - localPre) * t;

        G4double r = p.mag();
        if (r < fRMin || r >= fRMax) {
            // flush prev
            if (curBin >= 0 && curAccum > 0) {
                AccumulateHit(aStep, curAccum, curBin);
                anyHit = true;
            }
            curBin = -1; curAccum = 0;
            continue;
        }
        G4double theta = std::acos(std::max(-1.0, std::min(1.0, p.z()/std::max(r, 1e-30))));
        G4double thRel = theta - fSTheta;
        if (thRel < 0 || thRel >= fDTheta) {
            if (curBin >= 0 && curAccum > 0) {
                AccumulateHit(aStep, curAccum, curBin);
                anyHit = true;
            }
            curBin = -1; curAccum = 0;
            continue;
        }
        G4double phi = std::atan2(p.y(), p.x()) - fSPhi;
        while (phi < 0) phi += twopi;
        while (phi >= twopi) phi -= twopi;
        if (phi >= fDPhi) {
            if (curBin >= 0 && curAccum > 0) {
                AccumulateHit(aStep, curAccum, curBin);
                anyHit = true;
            }
            curBin = -1; curAccum = 0;
            continue;
        }

        G4int iR     = std::min(fNRBins - 1, std::max(0, int((r - fRMin) / binR)));
        G4int iPhi   = std::min(fNPhiBins - 1, std::max(0, int(phi / binPhi)));
        G4int iTheta = std::min(fNThetaBins - 1, std::max(0, int(thRel / binTheta)));

        // bin volume
        G4double rIn  = fRMin + iR * binR;
        G4double rOut = rIn + binR;
        G4double thIn  = fSTheta + iTheta * binTheta;
        G4double thOut = thIn + binTheta;
        G4double binVol = (rOut*rOut*rOut - rIn*rIn*rIn) / 3.0
                        * (std::cos(thIn) - std::cos(thOut)) * binPhi;
        if (binVol < 1e-30) continue;

        G4double contrib = (dseg / binVol) * weight;

        G4int binIdx = iR * fNPhiBins * fNThetaBins + iPhi * fNThetaBins + iTheta;

        if (binIdx == curBin) {
            curAccum += contrib;
        } else {
            if (curBin >= 0 && curAccum > 0) {
                AccumulateHit(aStep, curAccum, curBin);
                anyHit = true;
            }
            curBin = binIdx;
            curAccum = contrib;
        }
    }
    if (curBin >= 0 && curAccum > 0) {
        AccumulateHit(aStep, curAccum, curBin);
        anyHit = true;
    }
    return anyHit;
}
