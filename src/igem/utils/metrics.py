from avalanche.evaluation.metric_definitions import GenericPluginMetric


class ProjectionOverheadMetric(GenericPluginMetric):
    """
    Mean wall-clock time of a single projection (Appendix F, MPO).

    The plugin calls ``record`` once per projection. ``result`` returns the
    mean over the current experience; the run-level total and count are kept
    on the plugin itself so the exact ratio can be recovered afterwards.
    """

    def __init__(self):
        super().__init__(
            metric=None, reset_at="experience", emit_at="experience", mode="train"
        )
        self.elapsed = 0.0
        self.count = 0

    def record(self, elapsed: float):
        self.elapsed += elapsed
        self.count += 1

    def reset(self):
        self.elapsed = 0.0
        self.count = 0

    def result(self):
        if self.count == 0:
            return 0.0
        return self.elapsed / self.count

    def __str__(self):
        return "ProjectionOverhead"
