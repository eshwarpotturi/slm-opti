"""Builds tasks.json: 150 merchant chores, each with a small account snapshot
and the correct tool call(s). Everything is generated from a fixed seed, so the
set is reproducible. No real payments account or customer data is used.

Gold argument values:
  plain value            -> must match
  {"$absent_or": v}      -> argument must be missing, or equal v
  {"$absent": true}      -> argument must be missing (or null / empty)
"""
import json
import random
import string

R = random.Random(20261008)
TODAY = "2026-10-08"

NAMES = ["Ravi Kumar", "Priya Nair", "Arjun Mehta", "Sneha Reddy", "Karthik Iyer", "Ananya Das",
         "Vikram Singh", "Meera Pillai", "Rohan Gupta", "Emily Carter", "James Walker", "Sofia Rossi",
         "Lucas Meyer", "Hannah Schmidt", "Oliver Brown", "Chloe Martin", "Noah Wilson", "Isabella Garcia",
         "Ethan Clark", "Mia Thompson", "Daniel Lee", "Grace Hall", "Liam Turner", "Zoe Bennett"]
ITEMS = [("ceramic mug set", 34.0), ("leather wallet", 59.0), ("yoga mat", 42.5), ("desk lamp", 78.0),
         ("wireless earbuds", 129.0), ("linen shirt", 64.0), ("coffee grinder", 96.5), ("backpack", 88.0),
         ("phone case", 19.0), ("running shoes", 115.0), ("scented candle", 24.0), ("notebook pack", 16.5)]
SERVICES = [("logo design", 250.0), ("consulting hours", 80.0), ("website audit", 400.0),
            ("photo editing", 35.0), ("translation pages", 22.0), ("piano lessons", 45.0)]
PLANS = ["Monthly Coffee Box", "Pro Membership", "Weekly Meal Kit", "Premium Support"]
PRODUCTS = ["Starter Kit", "Gift Card", "Online Course", "Repair Service", "Travel Mug", "Sticker Pack"]
NEW_PRODUCTS = [("Bamboo Cutting Board", "PHYSICAL"), ("Photography Ebook", "DIGITAL"),
                ("Bike Tune-Up", "SERVICE"), ("Wool Scarf", "PHYSICAL"), ("Meditation Audio Pack", "DIGITAL"),
                ("Garden Design Session", "SERVICE"), ("Steel Water Bottle", "PHYSICAL"),
                ("Resume Template Bundle", "DIGITAL")]
CARRIERS = ["UPS", "FEDEX", "DHL", "USPS"]
AN = string.ascii_uppercase + string.digits


def rs(n, chars=AN):
    return "".join(R.choice(chars) for _ in range(n))


def email(name):
    return name.lower().replace(" ", ".") + "@example.com"


def account():
    c = R.sample(NAMES, 9)
    days = ["2026-10-07", "2026-10-06", "2026-10-05", "2026-10-03", "2026-10-01", "2026-09-28"]
    orders = []
    for i in range(5):
        item, price = R.choice(ITEMS)
        orders.append({"order_id": "5O" + rs(15), "capture_id": rs(17), "customer": c[i],
                       "email": email(c[i]), "date": R.choice(days), "item": item,
                       "amount": price, "currency": "USD",
                       "status": "REFUNDED" if i == 4 else "COMPLETED"})
    inv_status = ["DRAFT", "SENT", "SENT", "PAID"]
    invoices = []
    for k, i in enumerate(range(3, 7)):
        svc, price = R.choice(SERVICES)
        q = R.randint(1, 4)
        invoices.append({"invoice_id": "INV2-%s-%s-%s-%s" % (rs(4), rs(4), rs(4), rs(4)), "customer": c[i],
                         "email": email(c[i]), "item": svc, "amount": round(price * q, 2), "currency": "USD",
                         "status": inv_status[k]})
    states = ["REQUIRED_ACTION", "UNDER_REVIEW", "RESOLVED"]
    reasons = ["ITEM_NOT_RECEIVED", "NOT_AS_DESCRIBED", "UNAUTHORISED"]
    disputes = []
    for k, i in enumerate([0, 1, 4]):
        disputes.append({"dispute_id": "PP-R-%s-%s" % (rs(3, string.ascii_uppercase), rs(8, string.digits)),
                         "order_id": orders[i]["order_id"], "customer": c[i], "reason": reasons[k],
                         "amount": orders[i]["amount"], "state": states[k]})
    sub_status = ["ACTIVE", "ACTIVE", "CANCELLED"]
    subs = [{"subscription_id": "I-" + rs(12), "customer": c[i], "plan": R.choice(PLANS), "status": sub_status[k]}
            for k, i in enumerate(range(6, 9))]
    products = [{"product_id": "PROD-" + rs(17), "name": n} for n in R.sample(PRODUCTS, 3)]
    for lst in (orders, invoices, disputes, subs):
        R.shuffle(lst)
    return {"today": TODAY, "orders": orders, "invoices": invoices, "disputes": disputes,
            "subscriptions": subs, "products": products}


def pick(lst, **kw):
    out = [x for x in lst if all(x[k] == v for k, v in kw.items())]
    return out


def refundable(a):
    disputed = {d["order_id"] for d in a["disputes"]}
    return [o for o in a["orders"] if o["status"] == "COMPLETED" and o["order_id"] not in disputed]


def call(tool, **args):
    return {"tool": tool, "args": args}


# ---- atoms: each returns (sentence, [gold calls], key identifying the entity used) ----
def a_refund_full(a, used):
    o = choose(refundable(a), used, "order_id")
    p = R.choice(["Refund {c}'s order in full.", "{c} returned the {i}, give them all their money back.",
                  "Please issue a full refund to {c} for the {i}."])
    return p.format(c=o["customer"], i=o["item"]), [call("create_refund", capture_id=o["capture_id"],
            amount_value={"$absent_or": o["amount"]}, currency_code={"$absent_or": "USD"})]


def a_refund_partial(a, used):
    o = choose(refundable(a), used, "order_id")
    amt = R.choice([5, 8, 10, 12.5, 15])
    p = R.choice(["Refund ${m} to {c} for the {i} order, it arrived a bit scratched.",
                  "Give {c} ${m} back on the {i}, not the whole amount.",
                  "{c} was overcharged on shipping for the {i}. Refund ${m}."])
    return p.format(c=o["customer"], i=o["item"], m=fmt(amt)), [call("create_refund", capture_id=o["capture_id"],
            amount_value=amt, currency_code="USD")]


def a_remind(a, used):
    v = choose(pick(a["invoices"], status="SENT"), used, "invoice_id")
    p = R.choice(["Remind {c} about the unpaid invoice.", "{c} still hasn't paid the invoice I sent, nudge them.",
                  "Send {c} a payment reminder for the {i} invoice."])
    return p.format(c=v["customer"], i=v["item"]), [call("send_invoice_reminder", invoice_id=v["invoice_id"])]


def a_send_draft(a, used):
    v = choose(pick(a["invoices"], status="DRAFT"), used, "invoice_id")
    p = R.choice(["Send {c} the invoice I drafted.", "The draft invoice for {c} is ready, send it out.",
                  "Go ahead and send the {i} invoice to {c}, it is still in drafts."])
    return p.format(c=v["customer"], i=v["item"]), [call("send_invoice", invoice_id=v["invoice_id"])]


def a_cancel_inv(a, used):
    v = choose(pick(a["invoices"], status="SENT"), used, "invoice_id")
    p = R.choice(["Cancel the invoice I sent to {c}.", "{c} cancelled the {i} job, so cancel the invoice I sent them."])
    return p.format(c=v["customer"], i=v["item"]), [call("cancel_sent_invoice", invoice_id=v["invoice_id"])]


def a_record_payment(a, used):
    v = choose(pick(a["invoices"], status="SENT"), used, "invoice_id")
    m, word = R.choice([("BANK_TRANSFER", "bank transfer"), ("CASH", "cash"), ("CHECK", "check")])
    p = R.choice(["{c} paid the invoice in full by {w}, mark it as paid.",
                  "I received the full invoice amount from {c} by {w}. Record it."])
    return p.format(c=v["customer"], w=word), [call("record_payment_for_invoice", invoice_id=v["invoice_id"],
            amount_value=v["amount"], currency_code="USD", method=m)]


def a_qr(a, used):
    v = choose(pick(a["invoices"], status="SENT"), used, "invoice_id")
    p = R.choice(["Make a QR code for {c}'s invoice so they can scan and pay.", "I need a payment QR code for the invoice sent to {c}."])
    return p.format(c=v["customer"]), [call("generate_invoice_qr_code", invoice_id=v["invoice_id"])]


def a_cancel_sub(a, used):
    s = choose(pick(a["subscriptions"], status="ACTIVE"), used, "subscription_id")
    p = R.choice(["Cancel {c}'s subscription, they asked to stop.", "{c} no longer wants the {pl}. Cancel it."])
    return p.format(c=s["customer"], pl=s["plan"]), [call("cancel_subscription", subscription_id=s["subscription_id"])]


def a_accept_dispute(a, used):
    d = choose(pick(a["disputes"], state="REQUIRED_ACTION"), used, "dispute_id")
    p = R.choice(["Accept {c}'s dispute and let them have the refund.", "I don't want to fight {c}'s claim. Accept it."])
    return p.format(c=d["customer"]), [call("accept_dispute_claim", dispute_id=d["dispute_id"])]


def a_tracking(a, used):
    o = choose(pick(a["orders"], status="COMPLETED"), used, "order_id")
    car = R.choice(CARRIERS)
    tn = rs(2, string.ascii_uppercase) + rs(10, string.digits)
    names = {"UPS": "UPS", "FEDEX": "FedEx", "DHL": "DHL", "USPS": "USPS"}
    p = R.choice(["I shipped {c}'s order with {car}, tracking number {tn}. Add it.",
                  "Add tracking {tn} ({car}) to {c}'s {i} order."])
    return p.format(c=o["customer"], car=names[car], tn=tn, i=o["item"]), [call("create_shipment_tracking",
            order_id=o["order_id"], tracking_number=tn, carrier=car)]


def a_get_tracking(a, used):
    o = choose(pick(a["orders"], status="COMPLETED"), used, "order_id")
    return "Pull up the shipment tracking for {c}'s order.".format(c=o["customer"]), [call("get_shipment_tracking", order_id=o["order_id"])]


def a_sub_details(a, used):
    s = choose(a["subscriptions"], used, "subscription_id")
    return "Show me the details of {c}'s subscription.".format(c=s["customer"]), [call("show_subscription_details", subscription_id=s["subscription_id"])]


def a_dispute_details(a, used):
    d = choose(a["disputes"], used, "dispute_id")
    return "Open the full details of {c}'s dispute.".format(c=d["customer"]), [call("get_dispute", dispute_id=d["dispute_id"])]


def a_create_invoice(a, used):
    o = choose(a["orders"], used, "email")
    svc, price = R.choice(SERVICES)
    q = R.randint(1, 5)
    p = R.choice(["Bill {c} for {q} x {s} at ${m} each.", "Draft an invoice to {c}: {q} {s}, ${m} per unit."])
    return p.format(c=o["customer"], q=q, s=svc, m=fmt(price)), [call("create_invoice", recipient_email=o["email"],
            quantity=q, unit_amount=price, currency_code="USD")]


def a_list_disputes(a, used):
    p, g = R.choice([("Which disputes need my action?", "REQUIRED_ACTION"), ("Show me the disputes that are under review.", "UNDER_REVIEW"),
                     ("List the disputes that are already resolved.", "RESOLVED"), ("Show me all my disputes.", {"$absent": True})])
    return p, [call("list_disputes", dispute_state=g)]


def a_list_invoices(a, used):
    p, g = R.choice([("Which invoices have I sent that are still unpaid?", "SENT"), ("List my draft invoices.", "DRAFT"),
                     ("Which invoices have been paid?", "PAID"), ("Show the invoices I cancelled.", "CANCELLED"),
                     ("Show me all my invoices.", {"$absent": True})])
    return p, [call("list_invoices", status=g)]


def a_list_tx(a, used):
    p, s, e = R.choice([("Show me yesterday's transactions.", "2026-10-07", "2026-10-07"),
                        ("What transactions came in today?", "2026-10-08", "2026-10-08"),
                        ("List transactions from 1 October to 5 October.", "2026-10-01", "2026-10-05"),
                        ("Show all transactions in September.", "2026-09-01", "2026-09-30"),
                        ("Show the transactions for 3 October only.", "2026-10-03", "2026-10-03"),
                        ("List every transaction from 15 September to 30 September.", "2026-09-15", "2026-09-30")])
    return p, [call("list_transactions", start_date=s, end_date=e)]


def a_create_product(a, used):
    n, t = R.choice(NEW_PRODUCTS)
    word = {"PHYSICAL": "physical", "DIGITAL": "digital", "SERVICE": "service"}[t]
    p = R.choice(['Add a new {w} product called "{n}".', 'Create a {w} product in my catalogue named "{n}".'])
    return p.format(w=word, n=n), [call("create_product", name=n, type=t)]


def a_list_products(a, used):
    return R.choice(["What products do I have in my catalogue?", "List my products."]), [call("list_products")]


def a_list_plans(a, used):
    return R.choice(["What subscription plans do I offer?", "List my subscription plans."]), [call("list_subscription_plans")]


def choose(lst, used, key):
    lst = [x for x in lst if x[key] not in used and x.get("customer") not in used]
    if not lst:
        raise LookupError
    x = R.choice(lst)
    used.add(x[key])
    if "customer" in x:
        used.add(x["customer"])
    return x


def fmt(v):
    return ("%.2f" % v).rstrip("0").rstrip(".") if v != int(v) else str(int(v))


RESOLVE = [a_refund_full, a_refund_partial, a_remind, a_send_draft, a_cancel_inv, a_record_payment, a_qr,
           a_cancel_sub, a_accept_dispute, a_tracking, a_get_tracking, a_sub_details, a_dispute_details, a_create_invoice]
LISTING = [a_list_disputes, a_list_invoices, a_list_tx, a_create_product, a_list_products, a_list_plans]


def direct(a):
    k = R.choice(["inv", "order", "dispute", "sub", "refund", "product", "track", "qr"])
    if k == "inv":
        v = R.choice(a["invoices"]); return "Show me invoice %s." % v["invoice_id"], [call("get_invoice", invoice_id=v["invoice_id"])]
    if k == "order":
        o = R.choice(a["orders"]); return "Get the details of order %s." % o["order_id"], [call("get_order", order_id=o["order_id"])]
    if k == "dispute":
        d = R.choice(a["disputes"]); return "Open dispute %s." % d["dispute_id"], [call("get_dispute", dispute_id=d["dispute_id"])]
    if k == "sub":
        s = R.choice(a["subscriptions"]); return "Show subscription %s." % s["subscription_id"], [call("show_subscription_details", subscription_id=s["subscription_id"])]
    if k == "refund":
        r = rs(17); return "What is the status of refund %s?" % r, [call("get_refund", refund_id=r)]
    if k == "product":
        p = R.choice(a["products"]); return "Show me product %s." % p["product_id"], [call("show_product_details", product_id=p["product_id"])]
    if k == "track":
        o = R.choice(a["orders"]); return "Get the shipment tracking for order %s." % o["order_id"], [call("get_shipment_tracking", order_id=o["order_id"])]
    v = R.choice(a["invoices"]); return "Generate a QR code for invoice %s." % v["invoice_id"], [call("generate_invoice_qr_code", invoice_id=v["invoice_id"])]


def trap(a, kind):
    """Requests where the right move is to do nothing and ask. Gold is an empty list."""
    known = {x["customer"] for k in ("orders", "invoices", "disputes", "subscriptions") for x in a[k]}
    stranger = R.choice([n for n in NAMES if n not in known])
    if kind == "unknown":
        return R.choice(["Refund %s's order in full.", "Send %s a reminder about their unpaid invoice.", "Cancel %s's subscription."]) % stranger, "customer is not in the account"
    if kind == "refunded":
        o = pick(a["orders"], status="REFUNDED")[0]
        return "Refund %s's %s order in full." % (o["customer"], o["item"]), "order is already refunded"
    if kind == "paid":
        v = pick(a["invoices"], status="PAID")[0]
        return "Send %s a payment reminder for their invoice." % v["customer"], "invoice is already paid"
    if kind == "sub_cancelled":
        s = pick(a["subscriptions"], status="CANCELLED")[0]
        return "Cancel %s's subscription." % s["customer"], "subscription is already cancelled"
    if kind == "dispute_closed":
        d = R.choice([d for d in a["disputes"] if d["state"] != "REQUIRED_ACTION"])
        return "Accept %s's dispute claim." % d["customer"], "dispute is not awaiting the merchant's action"
    if kind == "ambiguous":
        o = refundable(a)[0]
        item, price = R.choice([x for x in ITEMS if x[0] != o["item"]])
        a["orders"].append({"order_id": "5O" + rs(15), "capture_id": rs(17), "customer": o["customer"], "email": o["email"],
                            "date": "2026-10-02", "item": item, "amount": price, "currency": "USD", "status": "COMPLETED"})
        R.shuffle(a["orders"])
        return "Refund %s's order in full." % o["customer"], "customer has two orders and the request does not say which"
    if kind == "unsupported":
        o = R.choice(a["orders"])
        return R.choice(["Change my payout bank account to the new one.", "Send $50 to my supplier from my balance.",
                         "Delete %s from my customer list." % o["customer"], "Update my business address to 14 Park Street.",
                         "Email %s a 10%% discount coupon." % o["customer"], "Raise the price of all my subscription plans by $2."]), "no tool can do this"
    if kind == "missing":
        o = pick(a["orders"], status="COMPLETED")[0]
        return R.choice(["Add shipment tracking to %s's order." % o["customer"], "Create an invoice for %s." % o["customer"],
                         "Give %s a partial refund on their order." % o["customer"]]), "a required detail is missing"
    raise ValueError(kind)


def build():
    tasks = []

    def add(cat, text, gold, a, why=None):
        t = {"id": "T%03d" % (len(tasks) + 1), "category": cat, "request": text, "account": a, "expected": gold}
        if why:
            t["why_no_action"] = why
        tasks.append(t)

    for _ in range(22):                                   # ID is given in the request
        a = account(); text, gold = direct(a); add("1_direct", text, gold, a)
    for i in range(42):                                   # must find the right record first
        a = account(); text, gold = RESOLVE[i % len(RESOLVE)](a, set()); add("2_lookup", text, gold, a)
    for i in range(24):                                   # lists, dates, creating things
        a = account(); text, gold = LISTING[i % len(LISTING)](a, set()); add("3_list_create", text, gold, a)
    n = 0
    while n < 36:                                         # two or three chores in one message
        a = account(); used = set(); k = 3 if n % 4 == 3 else 2
        fns = R.sample(RESOLVE, k - 1) + [R.choice(RESOLVE + LISTING)]
        try:
            parts = [f(a, used) for f in fns]
        except LookupError:
            continue
        if len({p[1][0]["tool"] for p in parts}) < len(parts):
            continue
        joiner = R.choice([" Also: ", " Next: ", " One more thing: "])
        text = parts[0][0] + "".join(joiner + p[0] for p in parts[1:])
        add("4_multi_step", text, [c for p in parts for c in p[1]], a); n += 1
    kinds = ["unknown", "refunded", "paid", "sub_cancelled", "dispute_closed", "ambiguous", "unsupported", "missing"]
    for i in range(26):                                   # should do nothing and ask
        a = account(); text, why = trap(a, kinds[i % len(kinds)]); add("5_should_not_act", text, [], a, why)
    return tasks


if __name__ == "__main__":
    tasks = build()
    json.dump(tasks, open("tasks.json", "w"), indent=1)
    from collections import Counter
    print(len(tasks), dict(Counter(t["category"] for t in tasks)))
