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
            },
            "action_id": "cb2d6ce9-6a13-4a08-93ac-da0903387210"
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

curd_operations = {
  "steps": [
    {
      "type": "ASK_USER",
      "action": "ask_name",
      "message": "Please enter the user's name."
    },
    {
      "type": "ASK_USER",
      "action": "ask_age",
      "message": "Please enter the user's age."
    },
    {
      "type": "API_CALL",
      "action": "create_user",
      "parameters": {
        "name": "<user_provided_name>",
        "age": "<user_provided_age>"
      },
      "action_id": "c8a99e10-9572-4696-aecd-ee2269a603fd"
    },
    {
      "type": "ASK_USER",
      "action": "ask_user_id",
      "message": "Please enter the user ID to fetch details."
    },
    {
      "type": "API_CALL",
      "action": "get_user",
      "parameters": {
        "user_id": "<user_provided_user_id>"
      },
      "action_id": "e1cc0241-5dd9-4d4f-bb92-e74b9fdafb49"
    },
    {
      "type": "RESPOND_FINAL",
      "message": "User creation and lookup process completed successfully!"
    }
  ]
}
