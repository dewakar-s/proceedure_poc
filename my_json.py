proceedure_json = {
    "steps": [
        {
            "type": "ASK_USER",
            "action": "ask_email",
            "message": "Please provide your email address to proceed with the order cancellation."
        },
        {
            "type": "API_CALL",
            "action": "fetch_orders",
            "parameters": {
                "email_id": "<user_provided_email>"
            }
        },
        {
            "type": "ASK_USER",
            "action": "select_order",
            "message": "Please select the order you wish to cancel from the list."
        },
        {
            "type": "API_CALL",
            "action": "cancel_order",
            "parameters": {
                "order_id": "<user_selected_order_id>"
            }
        },
        {
            "type": "RESPOND_FINAL",
            "message": "Your order has been successfully canceled. Thank you!"
        }
    ]
}

