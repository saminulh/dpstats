# Tasks

::: dpstats.task
    options:
      heading: Task contracts

::: dpstats.tasks
    options:
      heading: Task discovery

::: dpstats.tasks.m_estimation.losses
    options:
      heading: M-estimation loss properties
      members: [LossProperties]

::: dpstats.tasks.linear_regression.transformations.m_estimation
    options:
      heading: Residual losses
      members: [AbsoluteResidualLoss, HuberResidualLoss, RobHytResidualLoss, SquaredResidualLoss]

::: dpstats.tasks.mean_estimation.transformations.m_estimation
    options:
      heading: Location losses
      members: [HuberNormLoss, SquaredErrorLoss]

::: dpstats.tasks.median_estimation.transformations.to_m_estimation
    options:
      heading: Absolute-error loss
      members: [AbsoluteErrorLoss]
