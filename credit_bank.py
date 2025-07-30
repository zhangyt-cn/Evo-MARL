# credit_bank.py
import ray

@ray.remote
class CreditBank:
    def __init__(self, roles):
        self.scores = {r: 0.0 for r in roles}

    def add(self, role, delta):
        self.scores[role] += float(delta)
        return self.scores[role]

    def get(self, role):
        # 允许内部系统读取；但暴露给 agent 的接口用 get_others
        return self.scores[role]

    def get_others(self, role):
        return {r: v for r, v in self.scores.items() if r != role}

    def get_all(self):
        return dict(self.scores)

# singleton helper
_handle = None
def get_credit_bank(roles):
    global _handle
    if _handle is None:
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, include_dashboard=False)
        _handle = CreditBank.remote(roles)
    return _handle

