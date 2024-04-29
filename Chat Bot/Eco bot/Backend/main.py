from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import JSONResponse
import db_support
import generic_helper

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "hello"}


inprogress_orders = {}


@app.post('/')
async def handle_request(request: Request):
    #Retrive jason data from the request
    payload = await request.json()

    # Extract the necessary info from payload based on the structure of WebhookRequest from Dialogflow
    intent = payload['queryResult']['intent']['displayName']
    parameters = payload['queryResult']['parameters']
    output_context = payload['queryResult']['outputContexts']
    session_id = generic_helper.extract_session_id(output_context[0]['name'])
    default_fulfillment_text = payload['queryResult'].get('fulfillmentText')

    intent_handler_dict = {
        'new.order' : new_order_reset,
        'order.add - context: ongoing-order' : add_to_order,
        'order.remove - context: ongoing-order' : remove_from_order,
        'order.complete - context: ongoing-order': complete_order,
        'track.order - context: ongoing-tracking': track_order,
    }

    return intent_handler_dict[intent](parameters, session_id, default_fulfillment_text)


def new_order_reset(parameters: dict, session_id: str, default_fulfillment_text: str):
    if session_id in inprogress_orders:
        del inprogress_orders[session_id]
        
        return JSONResponse(content={
            "fulfillmentText": default_fulfillment_text
        })


def remove_from_order(parameters:dict, session_id: str, default_fulfillment_text: str):
    if session_id not in inprogress_orders:

        return JSONResponse(content={
            "fulfillmentText": "I'm having a trouble finding your order. Sorry! Can you place a new order please?"
    })
    
    current_order = inprogress_orders[session_id]
    food_items = parameters["food-item"]


    removed_items = []  # Track item removed
    no_such_items = []  # Item we didn't find
    for item in food_items:
        if item not in current_order:
            no_such_items.append(item)
        else:
            removed_items.append(item) 
            del current_order[item]

    if len(removed_items) > 0:
        fulfillment_text = f'Removed {",".join(removed_items)} from your order!'

    if len(no_such_items) > 0:
        fulfillment_text = f' Your current order does not have {",".join(no_such_items)}'

    if len(current_order.keys()) == 0:
        fulfillment_text += " Your order is empty!"
    else:
        order_str = generic_helper.get_str_frm_food_dict(current_order)
        fulfillment_text += f" Here is what is left in your order: {order_str}"

    return JSONResponse(content={
        "fulfillmentText": fulfillment_text
    })



def add_to_order(parameters:dict, session_id: str, default_fulfillment_text: str):
    food_items = parameters['food-item']
    quantity = parameters['number']

    if len(food_items) != len(quantity):
        fulfillment_text = 'Sorry, can you please specify the food items and quantity clearly'
    else:
        new_food_dict = dict(zip(food_items, quantity))  

        if session_id in inprogress_orders:
            current_food_dict = inprogress_orders[session_id]
            #current_food_dict.update(new_food_dict)
            merged_dict = {**current_food_dict, **new_food_dict}
            inprogress_orders[session_id] = merged_dict    
        else:
            inprogress_orders[session_id] = new_food_dict



        order_str = generic_helper.get_str_frm_food_dict(inprogress_orders[session_id])    
        fulfillment_text = f"so far you have: {order_str}, do you need anything else?"
        

    return JSONResponse(content = {
            'fulfillmentText' : fulfillment_text
    })  
    


def complete_order(parameters: dict, session_id: str, default_fulfillment_text: str):
    if session_id not in inprogress_orders:
        fulfillment_text = "I'm having trouble finding your order, Sorry.. can you place a new order please ?"
    else:
        order = inprogress_orders[session_id]
        order_id = save_to_db(order)

        if order_id == -1:
            fulfillment_text = "Sorry, I couldn't process your order due to a backend error. " \
                               "Please place a new order again"
        else:
            order_total = db_support.get_total_order_price(order_id)

            fulfillment_text = f"Awesome. We have placed your order. " \
                           f"Here is your order id # {order_id}. " \
                           f"Your order total is {order_total} which you can pay at the time of delivery!"
            
        del inprogress_orders[session_id]
    return JSONResponse(content = {
        "fulfillmentText": fulfillment_text
    })



def save_to_db(order: dict):
    # order = {'pizza':2, 'lassi': 1}
    next_order_id = db_support.get_next_order_id()

    for food_items, quantity in order.items():
        rcode = db_support.insert_order_item(
            food_items,
            quantity,
            next_order_id
        )

        if rcode == -1:
            return -1

    db_support.insert_order_tracking(next_order_id, "in progress")

    return next_order_id    



def track_order(parameters: dict, session_id: str, default_fulfillment_text: str):
    order_id = int(parameters['order_id'])
    order_status = db_support.get_order_status(order_id)


    if order_status:
        fulfillment_text = f"The Order Status for Order Id: {order_id} is {order_status}"
    else:
        fulfillment_text = f"No order found with order id: {order_id}"

    return JSONResponse(content = {
        "fulfillmentText": fulfillment_text
    })
    

     
