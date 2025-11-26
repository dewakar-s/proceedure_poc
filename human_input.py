from langgraph.types import interrupt

def human_call(state):
    # pause
    user_value = interrupt("wait_for_user_input")

    # resume when UI sends
    return user_value
