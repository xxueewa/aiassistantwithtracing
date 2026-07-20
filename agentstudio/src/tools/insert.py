import chromadb
from langchain_core.documents import Document
from services.vector_store_local import create_vector_store, upsert_documents

# Add documents
vs = create_vector_store()
upsert_documents(vs, [
    Document(page_content="Jordan Kim — VP, Asia-Pacific Sales. Home airport: SEA. Loyalty: Star Alliance Gold; Marriott Bonvoy Platinum. Seat: aisle; avoid bulkhead because of limited under-seat storage. " \
    "Food: pescatarian; no shellfish. Working style: wants a one-page executive brief, followed by detailed appendices. Communication: spoken morning briefing; written confirmation for all changes." \
    "Calendar: requires 45 minutes of preparation before customer negotiations.Approval authority: commercial terms up to $250,000; cannot approve security or legal exceptions.", metadata={"source": "terminal"}),
    Document(page_content="Liam O'Connor — Director, Infrastructure. Home airport: SEA. Loyalty: no airline preference. Seat: window; prefers the fewest connections. " \
    "Accessibility: needs step-free routes and an aisle chair available for aircraft boarding. Do not expose medical details in itinerary output. Food: vegetarian. " \
    "Working style: technical detail, architecture diagrams, and explicit open questions. Approval authority: technical design only.", metadata={"source": "terminal"}),
    Document(page_content='''Hannah Weiss — Director, Supply Operations. Home airport: AUS. 
             Loyalty: oneworld Sapphire. Seat: aisle. Prefers rail for journeys under 4 hours. 
             Food: no restrictions. Working style: concise incident timeline, owners, and deadlines. 
             Safety: requires documented facility induction before factory-floor access. 
             Approval authority: supplier recovery actions up to $100,000.''', metadata={"source":"terminal"}),
    Document(page_content='''Aisha Patel — VP, Investor Relations
            Home airport: SEA. Loyalty: Delta Platinum; Hilton Diamond.
            Seat: aisle. Prefers nonstop travel.
            Food: halal.
            Working style: message map, likely questions, and approved answers.
            Confidentiality: may use only board-approved or publicly disclosed financial information with investors.
            Approval authority: investor program and messaging within approved materials.''', metadata={"source":"terminal"}),
    Document(page_content='''Daniel Ortiz — CFO
            Home airport: SEA. Loyalty: Alaska MVP Gold; Hyatt Globalist.
            Seat: aisle. Prefers arrival the evening before board activity.
            Food: low sodium.
            Working style: financial summary first; decisions and exceptions highlighted.
            Accessibility: requires a quiet room for remote calls; this is a workplace requirement, not a medical disclosure.
            Approval authority: all finance and travel-policy exceptions.''', metadata={"source":"terminal"}),
    Document(page_content='''
            Location: Meridian Bank Tower, 8 Marina Boulevard, Singapore 018981
            Date: Tuesday, July 14, 2026
            Northstar attendees: Jordan Kim, Liam O'Connor
            Customer attendees: Evelyn Tan, Ravi Menon, Noor Aziz, Grace Lim
            Agenda
            Time	Topic	Owner	Desired outcome
            09:00-09:20	Executive introductions	Jordan / Evelyn	Confirm decision process
            09:20-10:15	Regional data architecture	Liam / Ravi	Agree deployment pattern
            10:15-10:30	Break	—	—
            10:30-11:15	Security and data residency	Noor / Liam	Resolve open controls
            11:15-12:00	Commercial structure	Jordan / Grace	Confirm pricing framework
            12:00-13:15	Hosted lunch	Evelyn	Relationship building
            13:15-14:00	Implementation milestones	Liam	Agree 90-day plan
            14:00-14:30	Decisions and next steps	Jordan	Assign owners and dates

            Preparation requirements
            Jordan needs 45 minutes of uninterrupted preparation before the meeting.
            Liam must have step-free access from hotel to meeting room and an accessible vehicle option.
            Bring the approved architecture diagram, security control matrix, and pricing summary.
            Do not circulate draft contract redlines in the general attendee packet.''', metadata={"source":"terminal"})
])

print("Documents inserted successfully.")