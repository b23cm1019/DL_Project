"""Import this before building a model so basicsr's registries know the ablation classes.

basicsr only auto-imports modules inside basicsr/archs, basicsr/models and basicsr/data, so the
ablation classes (kept outside basicsr on purpose) are registered by importing them here.
"""
import ablations.ablib.datasets  # noqa: F401  AblationConditionedPairedDataset, AblationSeverityDCPTPairedDataset
import ablations.ablib.networks  # noqa: F401  AblationFullNAFNet, AblationSeverityExtractor
import ablations.ablib.stage12  # noqa: F401  AblationSeverityPretrainModel, AblationSeverityAdaptationModel
