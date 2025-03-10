"""
Base Pipeline Module

This module provides the BasePipeline class, which is the foundation for all
pipeline implementations in the NexusML suite.
"""

import logging
from typing import Any, Dict, List, Optional

from nexusml.src.pipeline.context import PipelineContext
from nexusml.src.pipeline.stages.base import PipelineStage
from nexusml.src.utils.di.container import DIContainer

# Set up logging
logger = logging.getLogger(__name__)


class BasePipeline:
    """
    Base class for all pipelines.

    The BasePipeline class provides the foundation for all pipeline implementations
    in the NexusML suite. It manages a sequence of pipeline stages and coordinates
    their execution, providing a consistent execution context and error handling.

    Attributes:
        config: Configuration for the pipeline.
        container: DI container for resolving dependencies.
        stages: List of pipeline stages to execute.
    """

    def __init__(self, config: Dict[str, Any], container: DIContainer):
        """
        Initialize the base pipeline.

        Args:
            config: Configuration for the pipeline.
            container: DI container for resolving dependencies.
        """
        self.config = config
        self.container = container
        self.stages: List[PipelineStage] = []
        self._initialize_stages()

    def _initialize_stages(self) -> None:
        """
        Initialize the pipeline stages.

        This method should be overridden by subclasses to add stages to the pipeline.
        """
        pass

    def add_stage(self, stage: PipelineStage) -> None:
        """
        Add a stage to the pipeline.

        Args:
            stage: Stage to add.
        """
        self.stages.append(stage)
        logger.debug(f"Added stage {stage.__class__.__name__} to pipeline")

    def execute(self, **kwargs) -> PipelineContext:
        """
        Execute the pipeline.

        This method executes all stages in the pipeline in sequence, providing
        a consistent execution context and error handling.

        Args:
            **kwargs: Additional arguments for pipeline execution.

        Returns:
            Pipeline context with execution results.

        Raises:
            Exception: If any stage fails during execution.
        """
        context = PipelineContext()
        context.start()

        try:
            # Add pipeline configuration to context
            context.set("pipeline_config", self.config)
            context.set("kwargs", kwargs)

            # Execute each stage
            for i, stage in enumerate(self.stages):
                stage_name = stage.__class__.__name__
                logger.info(f"Executing stage {i+1}/{len(self.stages)}: {stage_name}")
                context.start_component(stage_name)

                try:
                    stage.execute(context, **kwargs)
                    context.end_component()
                except Exception as e:
                    logger.error(f"Error in stage {stage_name}: {str(e)}")
                    context.log("ERROR", f"Stage {stage_name} failed: {str(e)}")
                    context.end_component()
                    raise

            context.end("completed")
            logger.info("Pipeline execution completed successfully")
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            context.log("ERROR", f"Pipeline execution failed: {str(e)}")
            context.end("failed")
            raise

        return context
