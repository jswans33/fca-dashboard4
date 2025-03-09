"""
Tests for Base Pipeline Module

This module contains tests for the BasePipeline class, ensuring that
pipeline stages are properly executed and errors are handled correctly.
"""

import unittest
from unittest.mock import MagicMock, patch, call

from nexusml.core.di.container import DIContainer
from nexusml.core.pipeline.context import PipelineContext
from nexusml.core.pipeline.pipelines.base import BasePipeline
from nexusml.core.pipeline.stages.base import PipelineStage


class TestBasePipeline(unittest.TestCase):
    """
    Test case for BasePipeline class.
    """

    def setUp(self):
        """
        Set up the test case.
        """
        self.container = MagicMock(spec=DIContainer)
        self.config = {"test_key": "test_value"}

        # Create a concrete implementation of BasePipeline for testing
        class TestPipeline(BasePipeline):
            def _initialize_stages(self):
                pass

        self.pipeline = TestPipeline(self.config, self.container)

    def test_add_stage(self):
        """
        Test adding a stage to the pipeline.
        """
        # Create a mock stage
        stage = MagicMock(spec=PipelineStage)

        # Add the stage to the pipeline
        self.pipeline.add_stage(stage)

        # Verify that the stage was added
        self.assertEqual(len(self.pipeline.stages), 1)
        self.assertEqual(self.pipeline.stages[0], stage)

    def test_execute(self):
        """
        Test executing the pipeline.
        """
        # Create mock stages
        stage1 = MagicMock(spec=PipelineStage)
        stage2 = MagicMock(spec=PipelineStage)

        # Add the stages to the pipeline
        self.pipeline.add_stage(stage1)
        self.pipeline.add_stage(stage2)

        # Execute the pipeline
        context = self.pipeline.execute(test_arg="test_value")

        # Verify that the stages were executed
        stage1.execute.assert_called_once()
        stage2.execute.assert_called_once()

        # Verify that the context was started and ended
        self.assertEqual(context.status, "completed")

        # Verify that the pipeline configuration was added to the context
        self.assertEqual(context.get("pipeline_config"), self.config)
        self.assertEqual(context.get("kwargs"), {"test_arg": "test_value"})

    def test_execute_stage_error(self):
        """
        Test that errors during stage execution are properly handled.
        """
        # Create mock stages
        stage1 = MagicMock(spec=PipelineStage)
        stage2 = MagicMock(spec=PipelineStage)
        stage2.execute.side_effect = ValueError("Test error")
        stage3 = MagicMock(spec=PipelineStage)

        # Add the stages to the pipeline
        self.pipeline.add_stage(stage1)
        self.pipeline.add_stage(stage2)
        self.pipeline.add_stage(stage3)

        # Execute the pipeline and expect an error
        with self.assertRaises(ValueError):
            self.pipeline.execute()

        # Verify that only the first two stages were executed
        stage1.execute.assert_called_once()
        stage2.execute.assert_called_once()
        stage3.execute.assert_not_called()

    @patch('nexusml.core.pipeline.pipelines.base.PipelineContext')
    def test_execute_context_management(self, mock_context_class):
        """
        Test that the pipeline properly manages the context.
        """
        # Create a mock context
        mock_context = MagicMock(spec=PipelineContext)
        mock_context_class.return_value = mock_context

        # Create a mock stage
        stage = MagicMock(spec=PipelineStage)
        stage_name = stage.__class__.__name__

        # Add the stage to the pipeline
        self.pipeline.add_stage(stage)

        # Execute the pipeline
        self.pipeline.execute()

        # Verify that the context was properly managed
        mock_context.start.assert_called_once()
        mock_context.set.assert_any_call("pipeline_config", self.config)
        mock_context.set.assert_any_call("kwargs", {})
        mock_context.start_component.assert_called_once_with(stage_name)
        mock_context.end_component.assert_called_once()
        mock_context.end.assert_called_once_with("completed")

    @patch('nexusml.core.pipeline.pipelines.base.PipelineContext')
    def test_execute_error_handling(self, mock_context_class):
        """
        Test that the pipeline properly handles errors.
        """
        # Create a mock context
        mock_context = MagicMock(spec=PipelineContext)
        mock_context_class.return_value = mock_context

        # Create a mock stage that raises an error
        stage = MagicMock(spec=PipelineStage)
        stage_name = stage.__class__.__name__
        stage.execute.side_effect = ValueError("Test error")

        # Add the stage to the pipeline
        self.pipeline.add_stage(stage)

        # Execute the pipeline and expect an error
        with self.assertRaises(ValueError):
            self.pipeline.execute()

        # Verify that the context was properly managed
        mock_context.start.assert_called_once()
        mock_context.set.assert_any_call("pipeline_config", self.config)
        mock_context.set.assert_any_call("kwargs", {})
        mock_context.start_component.assert_called_once_with(stage_name)
        # Verify that log was called at least once (it's called twice in the implementation)
        mock_context.log.assert_called()
        self.assertEqual(mock_context.log.call_count, 2)
        mock_context.end_component.assert_called_once()
        mock_context.end.assert_called_once_with("failed")


class TestConcretePipelines(unittest.TestCase):
    """
    Test case for concrete pipeline implementations.
    """

    def setUp(self):
        """
        Set up the test case.
        """
        self.container = MagicMock(spec=DIContainer)
        self.config = {"test_key": "test_value"}

    def test_initialize_stages_called(self):
        """
        Test that _initialize_stages is called during initialization.
        """
        # Create a concrete implementation of BasePipeline for testing
        class TestPipeline(BasePipeline):
            def _initialize_stages(self):
                self.stages_initialized = True

        # Create a pipeline instance
        pipeline = TestPipeline(self.config, self.container)

        # Verify that _initialize_stages was called
        self.assertTrue(hasattr(pipeline, "stages_initialized"))
        self.assertTrue(pipeline.stages_initialized)


if __name__ == '__main__':
    unittest.main()