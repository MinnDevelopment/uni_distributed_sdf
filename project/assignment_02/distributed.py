from typing import List

from project import ProcessingNodeSensor, LinearSensor


class DistributedNodeSensor(ProcessingNodeSensor):
    def __init__(self, local: LinearSensor, remote: List[ProcessingNodeSensor]):
        super().__init__(local, [r.model for r in remote])

    def process(self, t):
        pass
