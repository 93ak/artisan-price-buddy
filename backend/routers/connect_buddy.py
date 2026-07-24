"""
Connect Buddy — FastAPI router
Insert this file at: backend/routers/connect_buddy.py

Wire it up in main.py the same way business_buddy is wired up:

    from routers import connect_buddy
    app.include_router(connect_buddy.router, prefix="/connect", tags=["connect-buddy"])

Prototype only: no database, no auth, no real booking/messaging. Action
endpoints just acknowledge the request and echo back a confirmation message
for the frontend to show as a popup.
"""

from fastapi import APIRouter
from pydantic import BaseModel

from data import connect_buddy_data
from services import connect_buddy_service

EXPERTS = connect_buddy_data.EXPERTS
CERTIFICATIONS = connect_buddy_data.CERTIFICATIONS
COLLABORATORS = connect_buddy_data.COLLABORATORS
COMMUNITIES = connect_buddy_data.COMMUNITIES
classify_query = connect_buddy_service.classify_query

# No prefix here — main.py applies "/connect" at include_router time,
# same pattern as business_buddy.router.
router = APIRouter()


# ── Bootstrap: hand the frontend every list once, up front ────────────────
@router.get("/bootstrap")
async def bootstrap():
    return {
        "experts": EXPERTS,
        "certifications": CERTIFICATIONS,
        "collaborators": COLLABORATORS,
        "communities": COMMUNITIES,
    }


# ── Smart matching: classify a free-text question, return matching cards ──
class AnalyzeRequest(BaseModel):
    message: str


def _score(entry_keywords: list[str], query_keywords: list[str]) -> int:
    entry_kw = {k.lower() for k in entry_keywords}
    query_kw = {k.lower() for k in query_keywords}
    score = 0
    for qk in query_kw:
        for ek in entry_kw:
            if qk in ek or ek in qk:
                score += 1
                break
    return score


@router.post("/analyze")
async def analyze(req: AnalyzeRequest):
    result = await classify_query(req.message)
    intent = result.get("intent", "none")
    keywords = result.get("keywords", [])
    reason = result.get("reason", "")

    pool_by_intent = {
        "expert": EXPERTS,
        "certification": CERTIFICATIONS,
        "collaboration": COLLABORATORS,
        "community": COMMUNITIES,
    }

    if intent == "none" or intent not in pool_by_intent:
        return {"intent": "none", "reason": reason, "matches": []}

    pool = pool_by_intent[intent]
    scored = [(entry, _score(entry.get("keywords", []), keywords)) for entry in pool]
    scored = [pair for pair in scored if pair[1] > 0]
    scored.sort(key=lambda pair: pair[1], reverse=True)

    matches = [entry for entry, _ in scored[:3]]
    # If nothing scored (LLM keywords too vague), just show the top 2 as a
    # gentle fallback rather than an empty result.
    if not matches:
        matches = pool[:2]

    return {"intent": intent, "reason": reason, "matches": matches}


# ── Mock action endpoints: all of these just return a canned confirmation ─
class ConsultationRequest(BaseModel):
    expert_id: str


@router.post("/request-consultation")
async def request_consultation(req: ConsultationRequest):
    expert = next((e for e in EXPERTS if e["id"] == req.expert_id), None)
    name = expert["name"] if expert else "the expert"
    return {"success": True, "message": f"Your consultation request has been sent to {name}. They'll reach out to confirm a time."}


class CertConsultRequest(BaseModel):
    certification_id: str


@router.post("/connect-certification-consultant")
async def connect_certification_consultant(req: CertConsultRequest):
    cert = next((c for c in CERTIFICATIONS if c["id"] == req.certification_id), None)
    name = cert["name"] if cert else "this certification"
    return {"success": True, "message": f"Request sent! A consultant for {name} will contact you shortly."}


class CommunityJoinRequest(BaseModel):
    community_id: str


@router.post("/join-community")
async def join_community(req: CommunityJoinRequest):
    community = next((c for c in COMMUNITIES if c["id"] == req.community_id), None)
    name = community["name"] if community else "the community"
    return {"success": True, "message": f"You've requested to join {name}. Look out for a welcome message soon."}


class IntroductionRequest(BaseModel):
    collaborator_id: str
    your_name: str = "A fellow artisan"
    your_craft: str = ""
    looking_for: str = ""


@router.post("/introduction-preview")
async def introduction_preview(req: IntroductionRequest):
    """Builds the editable preview text shown in the Smart Introduction modal."""
    collab = next((c for c in COLLABORATORS if c["id"] == req.collaborator_id), None)
    if not collab:
        return {"success": False, "message": "Couldn't find that collaborator."}

    craft = req.your_craft or collab["from_profession"]
    need = req.looking_for or collab["to_profession"].lower()
    message = (
        f"Hi {collab['contact_name']},\n"
        f"{req.your_name} is a {craft} looking for {need}. "
        f"Would you be interested in connecting?"
    )
    return {"success": True, "contact_name": collab["contact_name"], "preview": message}


class SendIntroductionRequest(BaseModel):
    collaborator_id: str
    message: str


@router.post("/send-introduction")
async def send_introduction(req: SendIntroductionRequest):
    collab = next((c for c in COLLABORATORS if c["id"] == req.collaborator_id), None)
    name = collab["contact_name"] if collab else "them"
    return {"success": True, "message": f"Introduction sent to {name}!"}