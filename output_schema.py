from typing import Dict, List, Tuple, Optional, Literal
from pydantic import BaseModel, Field

# Intent
class IntentBreakdown(BaseModel):
    queries: List[str] = Field(description="List of individual task queries extracted from the original query")

#Orders
class OrderItemAvailability(BaseModel):
    add: Dict[str, Tuple[int, float]] = Field(
        default_factory=dict,
        description="Items to add with quantity and price"
    )
    remove: Dict[str, Tuple[int, float]] = Field(
        default_factory=dict,
        description="Items to remove with quantity and price"
    )

class OrderItemOutput(BaseModel):
    available: OrderItemAvailability
    not_available: List[str] = Field(
        default_factory=list,
        description="Items requested but not present in the menu"
    )

#Customers
class CustomerInfo(BaseModel):
    name: Optional[str] = Field(
        default_factory=Optional[str],
        description="Name of the Customer"
    ),
    phone_number: Optional[str] = Field(
        default_factory=Optional[str],
        description="Contact of the Customer"
    )
    address: Optional[str] = Field(
        default_factory=Optional[str],
        description="Address of the Customer"
    )
    additional_details: Optional[str] = Field(
        default_factory=Optional[str],
        description="additional details"
    )

class CustomerDetailOutput(BaseModel):
    details: CustomerInfo
    response: str = Field(
        default_factory=str,
        description="Generated response"
    )

# ---------------- Booking / Reservation Models & Agents ---------------- #
class BookingDetails(BaseModel):
    """
    Core booking/reservation structure.
    date should be YYYY-MM-DD, time should be HH:MM (24h).
    """
    location: Optional[str] = Field(
        default="",
        description="Location of reservation"
    )
    date: Optional[str] = Field(
        default="",
        description="Booking date in YYYY-MM-DD format"
    )
    time: Optional[str] = Field(
        default="",
        description="Booking time in HH:MM (24h) format"
    )
    guests: Optional[int] = Field(
        default=None,
        description="Number of guests for reservation"
    )
    special_requests: Optional[str] = Field(
        default="",
        description="Any special requests (allergies, high-chair, accessibility, etc.)"
    )


class BookingDetailOutput(BaseModel):
    """
    Manager-style output: merged booking details + human response + completeness flag.
    Matches the pattern used for customer details in your Order flow.
    """
    details: BookingDetails
    response: str = Field(
        default_factory=str,
        description="Generated human-friendly response confirming status or asking for missing fields"
    )
    info_status: str = Field(
        default="incomplete",
        description="'complete' if key booking fields present else 'incomplete'"
    )


class BookingAvailabilityOutput(BaseModel):
    """
    Output schema for availability checks.
    """
    available: bool
    available_slots: List[str] = Field(
        default_factory=list,
        description="List of available alternative slots (date time strings) if any"
    )
    suggestion: Optional[str] = Field(
        default="",
        description="Short human suggestion when preferred slot unavailable"
    )

# ---------------- Booking Cancellation Models ---------------- #

class BookingCancellationDetails(BaseModel):
    booking_id: Optional[str] = Field(
        default="",
        description="Unique booking ID to cancel the reservation"
    )
    name: Optional[str] = Field(
        default="",
        description="Name under which the reservation was made"
    )
    phone_number: Optional[str] = Field(
        default="",
        description="Phone number used for the reservation"
    )
    reason: Optional[str] = Field(
        default="",
        description="Optional reason for cancellation"
    )


class BookingCancellationOutput(BaseModel):
    details: BookingCancellationDetails
    response: str = Field(
        default_factory=str,
        description="Human-friendly cancellation confirmation or missing-info request"
    )
    info_status: str = Field(
        default="incomplete",
        description="'complete' if enough info to cancel, else 'incomplete'"
    )

#COMPLAINT
# ---------------- Complaint Classification Schema ---------------- #
class ComplaintClassification(BaseModel):
    classification: Literal["Complaint", "Finish"] = Field(
        description="Complaint = ongoing, Finish = complaint completed"
    )
# ---------------- Complaint Updater ---------------- #
class ComplaintUpdateOutput(BaseModel):
    complain: str = Field(
        description="Merged complaint text combining the current complaint with the new query. If none, return an empty string."
    )
    response: str = Field(
        description="Follow-up question. If complaint updated, ask if they want to add more. If still empty, ask for details."
    )

class SupervisorDecision(BaseModel):
    next_node: Literal[
        "information",
        "order_supervisor",
        "customer_checker",
        "booking_supervisor",
        "complaint_supervisor",
    ]

class OrderSupervisorDecision(BaseModel):
    next_node: Literal["order_checker", "order_repeater", "order_start", "order_complete", "order_cancel"]

class BookingSupervisorDecision(BaseModel):
    next_node: Literal["booking_checker", "booking_repeater", "booking_start", "booking_complete", "booking_cancel"]

class ComplaintSupervisorDecision(BaseModel):
    next_node: Literal[
        "complaint_customer_check",
        "complaint_classifier",
        "complaint_save"
    ]