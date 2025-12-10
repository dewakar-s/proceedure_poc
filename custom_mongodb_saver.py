# custom_mongodb_saver.py

import pickle
import time
from typing import Optional, Iterator, Any, Sequence

from pymongo import MongoClient, ASCENDING, DESCENDING
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    CheckpointTuple,
    get_checkpoint_id,
)
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import ChannelVersions


class MongoDBSaver(BaseCheckpointSaver[str]):
    """
    MINIMAL MongoDB checkpointer for LangGraph 0.3.x

    Features:
    - Stores checkpoints
    - Stores pending writes
    - Supports resume
    - Fully compatible with LangGraph 0.3.x
    - Only ~120 lines (much smaller than full implementation)
    """

    def __init__(self, uri: str, db_name: str):
        super().__init__()
        client = MongoClient(uri)
        db = client[db_name]

        self._checkpoints = db["lg_checkpoints"]
        self._writes = db["lg_writes"]

        # Basic indexes for speed
        self._checkpoints.create_index(
            [("thread_id", ASCENDING), ("checkpoint_id", DESCENDING)]
        )
        self._writes.create_index(
            [("thread_id", ASCENDING), ("checkpoint_id", ASCENDING)]
        )

    # ---------------------------------------------------------
    # LOAD checkpoint
    # ---------------------------------------------------------
    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
            thread_id = config["configurable"]["thread_id"]
            checkpoint_id = get_checkpoint_id(config)

            q = {"thread_id": thread_id}

            if checkpoint_id:
                q["checkpoint_id"] = checkpoint_id
                doc = self._checkpoints.find_one(q)
            else:
                # Load latest checkpoint safely for PyMongo 4.x
                cursor = (
                    self._checkpoints.find({"thread_id": thread_id})
                    .sort("checkpoint_id", DESCENDING)
                    .limit(1)
                )
                try:
                    doc = cursor.next()
                except StopIteration:
                    doc = None

            if not doc:
                return None

            checkpoint = pickle.loads(doc["checkpoint_pickle"])
            metadata = pickle.loads(doc["metadata_pickle"])

            # Load writes
            writes_cursor = (
                self._writes.find({"thread_id": thread_id, "checkpoint_id": doc["checkpoint_id"]})
                .sort("idx", ASCENDING)
            )
            writes = [
            (
                w["task_id"],                     # REQUIRED by LangGraph
                w["channel"],
                pickle.loads(w["value_pickle"]),
            )
            for w in writes_cursor
            ]


            return CheckpointTuple(
                config={"configurable": {"thread_id": thread_id, "checkpoint_id": doc["checkpoint_id"]}},
                checkpoint=checkpoint,
                metadata=metadata,
                pending_writes=writes,
                parent_config=None,
            )


    # ---------------------------------------------------------
    # LIST checkpoints (minimal)
    # ---------------------------------------------------------
    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        thread_id = config["configurable"]["thread_id"]

        cur = (
            self._checkpoints.find({"thread_id": thread_id})
            .sort("checkpoint_id", DESCENDING)
        )

        for doc in cur:
            if limit is not None and limit <= 0:
                break
            limit = limit - 1 if limit else None

            checkpoint = pickle.loads(doc["checkpoint_pickle"])
            metadata = pickle.loads(doc["metadata_pickle"])

            yield CheckpointTuple(
                config={"configurable": {"thread_id": thread_id, "checkpoint_id": doc["checkpoint_id"]}},
                checkpoint=checkpoint,
                metadata=metadata,
                pending_writes=[],
                parent_config=None,
            )

    # ---------------------------------------------------------
    # SAVE checkpoint
    # ---------------------------------------------------------
    def put(
        self,
        config: RunnableConfig,
        checkpoint: dict,
        metadata: dict,
        new_versions: ChannelVersions,  # required argument
    ) -> RunnableConfig:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = checkpoint["id"]  # LangGraph generates IDs

        doc = {
            "thread_id": thread_id,
            "checkpoint_id": checkpoint_id,
            "checkpoint_pickle": pickle.dumps(checkpoint),
            "metadata_pickle": pickle.dumps(metadata),
            "ts": time.time(),
        }

        self._checkpoints.update_one(
            {"thread_id": thread_id, "checkpoint_id": checkpoint_id},
            {"$set": doc},
            upsert=True,
        )

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id,
            }
        }

    # ---------------------------------------------------------
    # SAVE writes (minimal implementation)
    # ---------------------------------------------------------
    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = config["configurable"]["checkpoint_id"]

        # get starting index
        base_idx = self._writes.count_documents(
            {"thread_id": thread_id, "checkpoint_id": checkpoint_id}
        )

        docs = []
        for i, (channel, value) in enumerate(writes):
            docs.append(
                {
                    "thread_id": thread_id,
                    "checkpoint_id": checkpoint_id,
                    "idx": base_idx + i,
                    "channel": channel,
                    "value_pickle": pickle.dumps(value),
                    "task_id": task_id,
                    "task_path": task_path,
                }
            )

        if docs:
            self._writes.insert_many(docs)

    # ---------------------------------------------------------
    # Delete thread state
    # ---------------------------------------------------------
    def delete_thread(self, thread_id: str) -> None:
        self._checkpoints.delete_many({"thread_id": thread_id})
        self._writes.delete_many({"thread_id": thread_id})

    # ---------------------------------------------------------
    # Version helper (simple)
    # ---------------------------------------------------------
    def get_next_version(self, current: Optional[str], channel) -> str:
        if current is None:
            return "1"
        try:
            return str(int(current) + 1)
        except:
            return "1"
