from mongoengine import (
    Document, IntField, StringField, FloatField, DateTimeField,
    ReferenceField, connect, disconnect
)
import uuid
from datetime import datetime

disconnect()
# 1) connect to your MongoDB (adjust host/db as needed)
connect(
    host="mongodb+srv://aliabdullah121:O3i23JPHZPS9sv5A@cluster0.aid5c3g.mongodb.net/restaurant?retryWrites=true&w=majority"
)

# 2) Define your models
class Customer(Document):
    meta = {"collection": "customers"}
    customer_id   = StringField(primary_key=True, required=True, default=lambda: str(uuid.uuid4()))
    name          = StringField(required=True, max_length=200)
    phone_number  = StringField(required=True, max_length=50)
    address       = StringField(required=True)

class BookingDoc(Document):
    meta = {"collection": "bookings"}

    booking_id   = StringField(
        primary_key=True,
        required=True,
        default=lambda: str(uuid.uuid4())
    )
    customer   = ReferenceField(Customer, required=True, reverse_delete_rule=2)

    location  = StringField(required=True, max_length=200)

    reservation_date = DateTimeField(required=True, default=datetime.utcnow)

    guests    = IntField(default = None)

    booking_placement_date = DateTimeField(required=True, default=datetime.utcnow)

    meta = {
        "collection": "bookings",
        "auto_create_index": False,
    }

class OrderDoc(Document):
    meta = {"collection": "orders"}

    order_id   = StringField(
        primary_key=True,
        required=True,
        default=lambda: str(uuid.uuid4())
    )
    customer   = ReferenceField(Customer, required=True, reverse_delete_rule=2)
    status     = StringField(
        choices=("complete", "incomplete"),     # only these two values allowed :contentReference[oaicite:0]{index=0}
        default="incomplete"                    # default to incomplete
    )
    order_date = DateTimeField(required=True, default=datetime.utcnow)

    total      = IntField(default=None)

    meta = {
        "collection": "orders",
        "auto_create_index": False,
    }


class OrderItem(Document):
    meta = {
        "collection": "order_items",
        "indexes": [
            # to ensure fast lookups and uniqueness per order+item
            {"fields": ("order", "item"), "unique": True}
        ]
    }
    order    = ReferenceField(OrderDoc, required=True, reverse_delete_rule=2)
    item     = StringField(required=True, max_length=200)
    quantity = IntField(required=True, min_value=1)
    price    = IntField(default=None)

class Complaint(Document):
    meta = {"collection": "complaints"}

    complaint_id = StringField(
        primary_key=True,
        required=True,
        default=lambda: str(uuid.uuid4())
    )
    customer = ReferenceField(Customer, required=True, reverse_delete_rule=2)
    complaint_text = StringField(required=True)
    complaint_date = DateTimeField(
        required=True,
        default=datetime.utcnow
    )