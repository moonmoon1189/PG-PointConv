import torch


class PriorityMapper:
    def __init__(self):
        pass

    def map_priority(self, risk_score):
        """
        risk_score: [N] tensor of continuous scores [0, 1]
        Returns 0-10 score and priority string
        """
        # Map to 0-10 scale
        scaled_score = risk_score * 10.0

        priorities = []
        for score in scaled_score:
            val = score.item()
            if val >= 7.5:
                priorities.append("High Intervention (Demolition/Leveling)")
            elif val >= 4.0:
                priorities.append("Intermediate Intervention (Handrails/Storage)")
            else:
                priorities.append("Low/Moderate Risk (No structural change)")

        return scaled_score, priorities