
class Scheduler:
    def order(self, candidates):
        return list(candidates)


class FIFOScheduler(Scheduler):
    def order(self, candidates):
        return sorted(candidates, key=lambda x: (x[0].hardware_thread_id, x[1].id))


class RRScheduler(Scheduler):
    def __init__(self):
        self.next_hardware_thread = 0

    def order(self, candidates):
        if not candidates:
            return []
        ids = [state.hardware_thread_id for state, _ in candidates]
        modulus = max(ids) + 1

        def key(item):
            state, instr = item
            return ((state.hardware_thread_id - self.next_hardware_thread) % modulus, instr.id)

        ordered = sorted(candidates, key=key)
        self.next_hardware_thread = (ordered[0][0].hardware_thread_id + 1) % modulus
        return ordered
