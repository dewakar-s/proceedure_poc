import os
from pymongo import MongoClient
from dotenv import load_dotenv
import uuid

load_dotenv()

MONGODB_ATLAS_URI = os.getenv("MONGODB_ATLAS_URI_actions")

client_actions = MongoClient(MONGODB_ATLAS_URI, uuidRepresentation="javaLegacy")

db_actions = client_actions["agentprod"]

def get_actions_collection(tenant_action_id):
    collection = db_actions['actions']

    # Convert to proper Legacy UUID (OMR format) used in DB
    actions = list(collection.find({ "_id" : tenant_action_id}))
    print("Fetched actions count:", len(actions))
    # actions = collection.find({"_id": "e4e62d57-cb09-4a73-976e-f4f5a2c35b38"})
    # print("Fetched:", actions)
    # print("Fetched:", len(actions))

    return actions

