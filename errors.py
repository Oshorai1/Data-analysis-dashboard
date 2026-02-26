class PipelineError(Exception):
    pass

class ValidationError(PipelineError):
    pass

class LoadError(PipelineError):
    pass
