# Work Chunk 11: Dependency Injection Integration

## Prompt

As the Infrastructure Specialist, your task is to update the core components to
use dependency injection. Currently, components create their dependencies
internally, leading to tight coupling and difficulty in testing. Your goal is to
update the components to use the dependency injection container, making them
more testable, configurable, and extensible.

## Context

The NexusML suite currently has several components that create their
dependencies internally:

- `EquipmentClassifier` creates its own `GenericFeatureEngineer` and
  `EAVManager`
- `GenericFeatureEngineer` creates its own `EAVManager`
- Various functions create their dependencies on demand

This tight coupling makes it difficult to:

- Test components in isolation
- Replace dependencies with alternative implementations
- Configure the system for different scenarios

By updating the components to use dependency injection, we can address these
issues while maintaining backward compatibility.

## Files to Create or Modify

1. **`nexusml/core/model.py`** (modify)

   - Update `EquipmentClassifier` to use dependency injection
   - Add constructor parameters for dependencies
   - Use the DI container for dependency resolution

2. **`nexusml/core/feature_engineering.py`** (modify)

   - Update `GenericFeatureEngineer` to use dependency injection
   - Add constructor parameters for dependencies
   - Use the DI container for dependency resolution

3. **`nexusml/core/di/decorators.py`**

   - Contains decorators for dependency injection
   - Includes `@inject` decorator for constructor injection
   - Includes `@injectable` decorator for class registration

4. **`nexusml/core/di/registration.py`**

   - Contains registration functions for the DI container
   - Registers all components with the container
   - Sets up default dependencies

5. **`tests/core/di/test_integration.py`**
   - Contains integration tests for DI
   - Tests component creation through the container
   - Tests dependency resolution

## Work Hierarchy

1. **Analysis Phase**

   - Review existing component dependencies
   - Identify injection points
   - Document backward compatibility requirements
   - Analyze testing requirements

2. **Design Phase**

   - Design dependency injection strategy
   - Design decorator interfaces
   - Design registration functions
   - Design backward compatibility strategy

3. **Implementation Phase**

   - Implement decorators
   - Implement registration functions
   - Update `EquipmentClassifier`
   - Update `GenericFeatureEngineer`
   - Update other components as needed

4. **Testing Phase**

   - Write unit tests for decorators
   - Write unit tests for registration functions
   - Write integration tests for DI
   - Test backward compatibility

5. **Documentation Phase**
   - Document dependency injection strategy
   - Create examples of using DI
   - Document migration from existing code
   - Document testing with DI

## Checklist

### Analysis

- [ ] Review `EquipmentClassifier` dependencies
- [ ] Review `GenericFeatureEngineer` dependencies
- [ ] Review other component dependencies
- [ ] Identify injection points
- [ ] Document backward compatibility requirements
- [ ] Analyze testing requirements

### Design

- [ ] Design dependency injection strategy
- [ ] Design `@inject` decorator
- [ ] Design `@injectable` decorator
- [ ] Design registration functions
- [ ] Design backward compatibility strategy
- [ ] Design testing strategy

### Implementation

- [ ] Implement `@inject` decorator
- [ ] Implement `@injectable` decorator
- [ ] Implement registration functions
- [ ] Update `EquipmentClassifier`
- [ ] Update `GenericFeatureEngineer`
- [ ] Update other components as needed
- [ ] Implement backward compatibility

### Testing

- [ ] Write unit tests for `@inject` decorator
- [ ] Write unit tests for `@injectable` decorator
- [ ] Write unit tests for registration functions
- [ ] Write integration tests for DI
- [ ] Test `EquipmentClassifier` with DI
- [ ] Test `GenericFeatureEngineer` with DI
- [ ] Test other components with DI
- [ ] Test backward compatibility

### Documentation

- [ ] Document dependency injection strategy
- [ ] Document `@inject` decorator
- [ ] Document `@injectable` decorator
- [ ] Document registration functions
- [ ] Create examples of using DI
- [ ] Document migration from existing code
- [ ] Document testing with DI
- [ ] Update main README with information about DI

## Dependencies

This work chunk depends on:

- Work Chunk 3: Dependency Injection Container
- Work Chunk 4: Data Components
- Work Chunk 5: Feature Engineering
- Work Chunk 6: Model Components
- Work Chunk 8: Pipeline Orchestrator

## Integration Points

- The updated components will use the DI container from Work Chunk 3
- The updated components will implement the interfaces from Work Chunk 2
- The updated components will be used by the orchestrator from Work Chunk 8

## Testing Criteria

- Components can be instantiated through the DI container
- Dependencies are properly injected
- Results match the existing pipeline
- Backward compatibility is maintained
- Unit tests pass for all updated components
- Integration tests pass for DI

## Definition of Done

- All checklist items are complete
- All tests pass
- Documentation is complete
- Code review has been completed
- Updated components follow SOLID principles
- Backward compatibility is maintained
- Components can be easily swapped or extended
